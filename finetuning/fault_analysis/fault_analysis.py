#!/usr/bin/env python
"""
Fault Analysis Script for TruFor Model
Evaluates the model on a dataset and outputs a CSV file with files that have Dice coefficient below 90%
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from tqdm import tqdm
from PIL import Image

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
from lib.config import config
from lib.utils import get_model

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Define paths based on the directory structure
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.masks_dir = os.path.join(self.root_dir, 'masks')
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
            
        print(f"Found {len(self.image_files)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
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
        
        return img_tensor, mask_tensor, img_path

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

def parse_args():
    parser = argparse.ArgumentParser(description='Fault Analysis for AI Manipulation Detection')
    parser.add_argument('--data_root', type=str, required=True, 
                      help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=1, 
                      help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, 
                      help='GPU ID')
    parser.add_argument('--output_csv', type=str, default='fault_analysis.csv', 
                      help='Output CSV file path')
    parser.add_argument('--threshold', type=float, default=0.9,
                      help='Threshold for fault detection (Dice coefficient)')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model for evaluation')
    parser.add_argument('--np_weights', type=str, 
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/noiseprint++/noiseprint++.th',
                      help='Path to Noiseprint++ weights')
    parser.add_argument('--pretrained', type=str,
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth',
                      help='Path to pretrained backbone weights')
    parser.add_argument('--pred_threshold', type=float, default=0.5,
                      help='Threshold for binary prediction')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
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
    print(f"Model path: {args.model_path}")
    print(f"Dice coefficient threshold for fault: {args.threshold}")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    try:
        test_dataset = TestDataset(args.data_root)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        sys.exit(1)
    
    # Create data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
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
    
    # Load the trained model
    print(f"Loading model from {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Prepare CSV file
    csv_file = open(args.output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['filename', 'dice_coefficient', 'status'])
    
    # Evaluation
    print("Running fault analysis...")
    model.eval()
    fault_count = 0
    success_count = 0
    
    progress_bar = tqdm(test_loader, desc="Analyzing")
    with torch.no_grad():
        for images, masks, img_paths in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs, _, _, _ = model(images)
            
            # Calculate Dice coefficient
            pred = outputs[:, 1]  # Class 1 is "manipulated"
            
            for i in range(len(images)):
                dice = dice_coefficient(pred[i], masks[i].float(), threshold=args.pred_threshold)
                if isinstance(dice, torch.Tensor):
                    dice_value = dice.item()
                else:
                    dice_value = dice
                
                filename = os.path.basename(img_paths[i])
                status = "SUCCESS" if dice_value >= args.threshold else "FAULT"
                
                if dice_value < args.threshold:
                    fault_count += 1
                else:
                    success_count += 1
                
                # Write to CSV
                csv_writer.writerow([filename, f"{dice_value:.4f}", status])
    
    csv_file.close()
    
    print(f"Fault analysis complete!")
    print(f"Total images: {len(test_dataset)}")
    print(f"Success count (Dice >= {args.threshold}): {success_count}")
    print(f"Fault count (Dice < {args.threshold}): {fault_count}")
    print(f"Fault rate: {fault_count / len(test_dataset) * 100:.2f}%")
    print(f"Results saved to {args.output_csv}")

if __name__ == '__main__':
    main()