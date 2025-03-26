#!/usr/bin/env python
"""
Fine-tuning Script for AI Manipulation Detection using TruFor
Continues training from a pretrained model checkpoint
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import wandb

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
from lib.config import config
from lib.utils import get_model

class AIManipDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train', val_ratio=0.2):
        self.root_dir = root_dir
        self.split = split
        self.val_ratio = val_ratio
        
        # Define paths based on the directory structure
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.masks_dir = os.path.join(self.root_dir, 'masks')
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        # Fix random seed for reproducibility
        np.random.seed(42)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
            
        # Create split based on validation ratio
        if split == 'valid':
            indices = np.random.permutation(len(self.image_files))
            valid_size = max(1, int(self.val_ratio * len(self.image_files)))
            self.image_files = [self.image_files[i] for i in indices[:valid_size]]
        elif split == 'train':
            indices = np.random.permutation(len(self.image_files))
            valid_size = max(1, int(self.val_ratio * len(self.image_files)))
            self.image_files = [self.image_files[i] for i in indices[valid_size:]]
        # For test, use all data
        
        print(f"Found {len(self.image_files)} images for {split} in {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def shuffle(self):
        np.random.shuffle(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, name + ext)
        if not os.path.exists(mask_path):
            # Try other extensions
            for ext_try in ['.png', '.jpg', '.jpeg']:
                mask_path = os.path.join(self.masks_dir, name + ext_try)
                if os.path.exists(mask_path):
                    break
        
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            mask_tensor = torch.from_numpy((np.array(mask) > 127).astype(np.int64))
        else:
            # If no mask is found, assume the entire image is manipulated
            mask_tensor = torch.ones((image.height, image.width), dtype=torch.int64)
        
        return img_tensor, mask_tensor

def dice_coefficient(pred, target, smooth=1.0, threshold=0.5):
    """
    Calculate Dice coefficient
    
    Args:
        pred: Prediction tensor (B, H, W) or (H, W)
        target: Target tensor (B, H, W) or (H, W)
        smooth: Smoothing factor
        threshold: Threshold for binary prediction
        
    Returns:
        Dice coefficient (float)
    """
    if isinstance(pred, torch.Tensor):
        # Apply sigmoid if prediction is logits
        if pred.max() > 1 or pred.min() < 0:
            pred = torch.sigmoid(pred)
        
        # Apply threshold
        pred = (pred > threshold).float()
        target = target.float()
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Handle empty masks
        if union == 0:
            return 1.0
        
        # Return Dice coefficient
        return (2. * intersection + smooth) / (union + smooth)
    else:
        # For numpy arrays
        pred = (pred > threshold).astype(np.float32)
        target = target.astype(np.float32)
        
        # Flatten arrays
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Handle empty masks
        if union == 0:
            return 1.0
        
        # Return Dice coefficient
        return (2. * intersection + smooth) / (union + smooth)

def dice_loss(pred, target, smooth=1.0):
    """
    Calculate Dice loss
    """
    dice = dice_coefficient(pred, target, smooth)
    return 1.0 - dice

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.criterion(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def train(model, train_loader, optimizer, device, epoch, num_epochs):
    """
    Train model for one epoch
    
    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0.0
    focal_criterion = FocalLoss(alpha=0.25, gamma=2)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs, _, _, _ = model(images)
        
        # Modified loss: Dice loss + Focal loss
        dl = dice_loss(outputs[:, 1], masks.float())
        fl = focal_criterion(outputs, masks)
        loss = dl + fl
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return train_loss / len(train_loader)

def evaluate(model, val_loader, device, threshold=0.5):
    """
    Evaluate model on validation set
    
    Returns:
        Average validation loss, average Dice coefficient
    """
    model.eval()
    val_loss = 0.0
    dice_scores = []
    focal_criterion = FocalLoss(alpha=0.25, gamma=2)
    
    progress_bar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs, _, _, _ = model(images)
            
            # Compute loss (same as training)
            dl = dice_loss(outputs[:, 1], masks.float())
            fl = focal_criterion(outputs, masks)
            loss = dl + fl
            
            val_loss += loss.item()
            
            # Calculate Dice coefficient
            pred = outputs[:, 1]  # Class 1 is "manipulated"
            dice = dice_coefficient(pred, masks.float(), threshold=threshold)
            
            # Handle both tensor and float returns
            if isinstance(dice, torch.Tensor):
                dice_scores.append(dice.item())
            else:
                dice_scores.append(dice)  # Already a float
            
            progress_bar.set_postfix({'loss': loss.item(), 'dice': dice_scores[-1]})
    
    return val_loss / len(val_loader), np.mean(dice_scores)

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune AI Manipulation Detection model')
    parser.add_argument('--data_root', type=str, required=True, 
                      help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=64, 
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, 
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-6, 
                      help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, 
                      help='GPU ID')
    parser.add_argument('--output_dir', type=str, default='output_finetune', 
                      help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, 
                      help='Validation set ratio')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for binary prediction')
    parser.add_argument('--np_weights', type=str, 
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/noiseprint++/noiseprint++.th',
                      help='Path to Noiseprint++ weights')
    parser.add_argument('--pretrained', type=str,
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth',
                      help='Path to pretrained backbone weights')
    parser.add_argument('--model_path', type=str, 
                      default='/root/cv_hack_25/TruFor/finetuning/archive/output_sub_2/best_dice.pth',
                      help='Path to previously trained model for fine-tuning')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    CONFIG = {
        "architecture": "TruFor",
        "encoder": "mit_b2",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss": "DiceLoss + FocalLoss",  
        "optimizer": "Adam",
        "dataset": "train_full_combined",
    }
    
    # Check if data directory exists
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data directory {args.data_root} does not exist")
        sys.exit(1)
    
    # Verify model file exists
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file {args.model_path} does not exist")
        sys.exit(1)
    
    # Verify weight files exist
    if not os.path.exists(args.np_weights):
        print(f"ERROR: Cannot find Noiseprint++ weights at {args.np_weights}")
        sys.exit(1)
    
    if not os.path.exists(args.pretrained):
        print(f"ERROR: Cannot find pretrained weights at {args.pretrained}")
        sys.exit(1)
    
    print(f"Using dataset: {args.data_root}")
    print(f"Fine-tuning from model: {args.model_path}")
    print(f"Noiseprint++ weights: {args.np_weights}")
    print(f"Pretrained weights: {args.pretrained}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of epochs: {args.epochs}")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    try:
        train_dataset = AIManipDataset(args.data_root, split='train', val_ratio=args.val_ratio)
        val_dataset = AIManipDataset(args.data_root, split='valid', val_ratio=args.val_ratio)
    except Exception as e:
        print(f"Error creating datasets: {e}")
        sys.exit(1)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    # Create model - Use TruFor's config for model creation
    config.defrost()
    
    # Basic settings
    config.MODEL.NAME = 'detconfcmx'
    config.MODEL.MODS = ('RGB','NP++')
    config.MODEL.PRETRAINED = args.pretrained
    config.DATASET.NUM_CLASSES = 2
    
    # Extra settings
    config.MODEL.EXTRA.BACKBONE = 'mit_b2'
    config.MODEL.EXTRA.DECODER = 'MLPDecoder'
    config.MODEL.EXTRA.DECODER_EMBED_DIM = 512
    config.MODEL.EXTRA.PREPRC = 'imagenet'
    config.MODEL.EXTRA.NP_WEIGHTS = args.np_weights
    config.MODEL.EXTRA.MODULES = ['NP++','backbone','loc_head','conf_head','det_head']
    config.MODEL.EXTRA.FIX_MODULES = ['NP++']
    config.MODEL.EXTRA.DETECTION = 'confpool'
    config.MODEL.EXTRA.CONF = True
    
    # Missing parameters
    config.MODEL.EXTRA.BN_EPS = 0.001
    config.MODEL.EXTRA.BN_MOMENTUM = 0.1
    
    config.freeze()
    
    print("Creating model...")
    try:
        model = get_model(config)
        model = model.to(device)
        print("Model created successfully!")
    except Exception as e:
        print(f"Error creating model: {e}")
        sys.exit(1)
    
    # Load the pretrained model
    print(f"Loading pretrained model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            initial_dice = checkpoint.get('dice', 0.0)
            print(f"Model loaded successfully! Previous best Dice: {initial_dice:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    wandb.init(project="hackathon-truefor", config=CONFIG)

    # First evaluate the model before training
    print("Evaluating model before training...")
    val_loss, dice_score = evaluate(model, val_loader, device, args.threshold)
    print(f"Before training - Val Loss: {val_loss:.4f}, Dice: {dice_score:.4f}")
    wandb.log({"initial_val_loss": val_loss, "initial_dice": dice_score})
    
    # Get best dice from the model or use the current one if not available
    try:
        best_dice = checkpoint.get('dice', dice_score)
    except:
        best_dice = dice_score
    
    print(f"Starting fine-tuning with best dice: {best_dice:.4f}")
    
    # Training loop - fine-tune for a few epochs
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train(model, train_loader, optimizer, device, epoch, args.epochs)
        
        # Evaluate
        val_loss, dice_score = evaluate(model, val_loader, device, args.threshold)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {dice_score:.4f}")
        wandb.log({
            'epoch': epoch, 
            'train_loss': train_loss, 
            'val_loss': val_loss, 
            'dice_score': dice_score
        })
        
        # Save best model based on dice coefficient
        if dice_score > best_dice:
            best_dice = dice_score
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dice': best_dice,
            }, os.path.join(args.output_dir, 'best_dice.pth'))
            print(f"Saved best model with Dice: {best_dice:.4f}")
        
        # Save checkpoint after each epoch (for this short training)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'dice': dice_score,
        }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dice': dice_score,
    }, os.path.join(args.output_dir, 'model_final.pth'))
    
    print(f'Fine-tuning complete! Best Dice: {best_dice:.4f}')
    wandb.log({
        'final_dice': best_dice,
        'dice_improvement': best_dice - dice_score  # Improvement from initial score
    })
    wandb.finish()

if __name__ == '__main__':
    main()