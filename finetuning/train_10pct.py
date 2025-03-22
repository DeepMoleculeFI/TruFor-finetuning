#!/usr/bin/env python
"""
Training script for AI Manipulation Detection using TruFor
Modified to use a smaller dataset for faster training
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
from lib.config import config
from lib.utils import get_model

# Import the AIManipDataset class or copy it here
class AIManipDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        
        # Define paths based on the directory structure
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.masks_dir = os.path.join(self.root_dir, 'masks')
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        if split == 'valid':
            # Use 20% of data for validation
            np.random.seed(42)
            indices = np.random.permutation(len(self.image_files))
            valid_size = int(0.2 * len(self.image_files))
            self.image_files = [self.image_files[i] for i in indices[:valid_size]]
        elif split == 'train':
            # Use 80% of data for training
            np.random.seed(42)
            indices = np.random.permutation(len(self.image_files))
            valid_size = int(0.2 * len(self.image_files))
            self.image_files = [self.image_files[i] for i in indices[valid_size:]]
        elif split == 'test':
            # Use all data for testing
            pass
        
        print(f"Found {len(self.image_files)} images for {split}")
    
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

def parse_args():
    parser = argparse.ArgumentParser(description='Train TruFor for AI Manipulation Detection')
    parser.add_argument('--data_root', type=str, default='/root/cv_hack_25/data/train_10pct', 
                      help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--output_dir', type=str, default='output_10pct', help='Output directory')
    parser.add_argument('--np_weights', type=str, 
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/noiseprint++/noiseprint++.th',
                      help='Path to Noiseprint++ weights')
    parser.add_argument('--pretrained', type=str,
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth',
                      help='Path to pretrained backbone weights')
    
    return parser.parse_args()

def dice_loss(pred, target, smooth=1.0):
    """
    Calculate Dice loss
    """
    pred = torch.sigmoid(pred)
    
    # Flatten predictions and targets
    pred_flat = pred.view(-1)
    target_flat = target.view(-1).float()
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Return Dice loss
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verify the weight files exist
    if not os.path.exists(args.np_weights):
        print(f"ERROR: Cannot find Noiseprint++ weights at {args.np_weights}")
        sys.exit(1)
    
    if not os.path.exists(args.pretrained):
        print(f"ERROR: Cannot find pretrained weights at {args.pretrained}")
        sys.exit(1)
        
    print(f"Noiseprint++ weights: {args.np_weights}")
    print(f"Pretrained weights: {args.pretrained}")
        
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = AIManipDataset(args.data_root, split='train')
    val_dataset = AIManipDataset(args.data_root, split='valid')
    
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
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create loss functions
    ce_criterion = nn.CrossEntropyLoss()
    
    # Calculate dice coefficient in validation
    def calculate_dice(pred, target, threshold=0.5):
        pred = torch.sigmoid(pred)
        pred_binary = (pred > threshold).float()
        
        # Handle empty masks
        if target.sum() == 0 and pred_binary.sum() == 0:
            return 1.0
            
        # Flatten predictions and targets
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1).float()
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Return Dice coefficient
        return (2. * intersection) / (union + 1e-6)
    
    # Training loop
    best_dice = 0.0
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs, _, _, _ = model(images)
            
            # Cross entropy loss
            loss = ce_criterion(outputs, masks)
            
            # Add dice loss component
            dice = dice_loss(outputs[:, 1], masks.float())
            combined_loss = loss + dice
            
            combined_loss.backward()
            optimizer.step()
            
            train_loss += combined_loss.item()
            progress_bar.set_postfix({'loss': combined_loss.item()})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        dice_scores = []
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]")
        
        with torch.no_grad():
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs, _, _, _ = model(images)
                
                # Calculate loss
                loss = ce_criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate Dice coefficient
                dice = calculate_dice(outputs[:, 1], masks.float())
                dice_scores.append(dice.item())
                
                progress_bar.set_postfix({'loss': loss.item(), 'dice': dice.item()})
        
        # Calculate average validation loss and dice
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = sum(dice_scores) / len(dice_scores)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}")
        
        # Save best model based on dice coefficient
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dice': best_dice,
            }, f'{args.output_dir}/best_dice.pth')
            print(f"Saved best model with Dice: {best_dice:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dice': avg_dice,
            }, f'{args.output_dir}/checkpoint_epoch{epoch+1}.pth')
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'dice': avg_dice,
    }, f'{args.output_dir}/model_final.pth')
    
    print(f'Training complete! Best Dice: {best_dice:.4f}')

if __name__ == '__main__':
    # Import these here to avoid path issues
    import glob
    import numpy as np
    from PIL import Image
    
    main()