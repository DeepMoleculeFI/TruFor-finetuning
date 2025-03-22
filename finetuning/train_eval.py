#!/usr/bin/env python
"""
Combined Training and Evaluation Script for AI Manipulation Detection using TruFor
Supports training on any dataset size and evaluates using Dice coefficient
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

def train(model, train_loader, optimizer, device, epoch, num_epochs):
    """
    Train model for one epoch
    
    Returns:
        Average training loss
    """
    model.train()
    train_loss = 0.0
    ce_criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs, _, _, _ = model(images)
        
        # Cross entropy loss
        ce_loss = ce_criterion(outputs, masks)
        
        # Dice loss
        dice = dice_loss(outputs[:, 1], masks.float())
        
        # Combined loss
        loss = ce_loss + dice
        
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
    ce_criterion = nn.CrossEntropyLoss()
    
    progress_bar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs, _, _, _ = model(images)
            
            # Cross entropy loss
            loss = ce_criterion(outputs, masks)
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
    parser = argparse.ArgumentParser(description='Train and Evaluate AI Manipulation Detection')
    parser.add_argument('--data_root', type=str, required=True, 
                      help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, 
                      help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, 
                      help='GPU ID')
    parser.add_argument('--output_dir', type=str, default='output', 
                      help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, 
                      help='Validation set ratio')
    parser.add_argument('--eval_only', action='store_true',
                      help='Run evaluation only')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to trained model for evaluation')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for binary prediction')
    parser.add_argument('--np_weights', type=str, 
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/noiseprint++/noiseprint++.th',
                      help='Path to Noiseprint++ weights')
    parser.add_argument('--pretrained', type=str,
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth',
                      help='Path to pretrained backbone weights')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data directory {args.data_root} does not exist")
        sys.exit(1)
    
    # Verify weight files exist
    if not os.path.exists(args.np_weights):
        print(f"ERROR: Cannot find Noiseprint++ weights at {args.np_weights}")
        sys.exit(1)
    
    if not os.path.exists(args.pretrained):
        print(f"ERROR: Cannot find pretrained weights at {args.pretrained}")
        sys.exit(1)
    
    print(f"Using dataset: {args.data_root}")
    print(f"Noiseprint++ weights: {args.np_weights}")
    print(f"Pretrained weights: {args.pretrained}")
    
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
        val_dataset, batch_size=1, shuffle=False, num_workers=4
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
    
    # Evaluation only mode
    if args.eval_only:
        if args.model_path is None:
            print("ERROR: --model_path must be specified in evaluation mode")
            sys.exit(1)
        
        if not os.path.exists(args.model_path):
            print(f"ERROR: Model file {args.model_path} does not exist")
            sys.exit(1)
        
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
        
        # Run evaluation
        val_loss, dice_score = evaluate(model, val_loader, device, args.threshold)
        print(f"Validation Loss: {val_loss:.4f}, Dice Coefficient: {dice_score:.4f}")
        
        return
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train(model, train_loader, optimizer, device, epoch, args.epochs)
        
        # Evaluate
        val_loss, dice_score = evaluate(model, val_loader, device, args.threshold)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Dice: {dice_score:.4f}")
        
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
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
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
    
    print(f'Training complete! Best Dice: {best_dice:.4f}')

if __name__ == '__main__':
    main()