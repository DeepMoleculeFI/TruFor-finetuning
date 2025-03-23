#!/usr/bin/env python
"""
Script to visualize 100 random images and their predicted masks
Saves both the original images and color-overlay masks to a specified output folder
"""

import os
import sys
import glob
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import time

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
from lib.config import config
from lib.utils import get_model

def load_images(image_paths, img_size=None):
    """Load and preprocess a batch of images in parallel"""
    images = []
    original_images = []
    
    def load_image(path):
        image = Image.open(path).convert('RGB')
        # Save original image
        orig_img = np.array(image)
        # Resize for model if needed
        if img_size:
            image = image.resize(img_size, Image.BILINEAR)
        img_array = np.array(image)
        return img_array, orig_img
    
    # Use thread pool for parallel loading
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(load_image, image_paths))
    
    # Unpack results
    images = [r[0] for r in results]
    original_images = [r[1] for r in results]
    
    # Stack images into a batch
    batch = np.stack(images)
    # Convert to tensor [B, H, W, C] -> [B, C, H, W]
    batch_tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
    
    return batch_tensor, original_images

def process_batch(model, image_batch, device, threshold=0.5):
    """Process a batch of images"""
    with torch.no_grad():
        # Move batch to device
        image_batch = image_batch.to(device)
        
        # Run inference
        outputs, _, _, _ = model(image_batch)
        
        # Get prediction maps (class 1 is "manipulated")
        preds = F.softmax(outputs, dim=1)[:, 1]
        
        # Return raw predictions and thresholded masks
        pred_np = preds.cpu().numpy()
        binary_masks = (pred_np > threshold).astype(np.uint8)
        
    return pred_np, binary_masks

def create_overlay(image, mask, alpha=0.5):
    """Create a visualization with the mask overlaid on the image"""
    # Create a copy of the image
    overlay = image.copy()
    
    # Create a colored mask (red for manipulated areas)
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [255, 0, 0]  # Red color for manipulated regions
    
    # Blend the image and mask
    cv_overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    return cv_overlay

def save_visualization(original_image, pred_mask, output_path, img_name, gt_mask=None, alpha=0.4):
    """Save original image and mask visualization"""
    # Save original image
    Image.fromarray(original_image).save(os.path.join(output_path, f"{img_name}.png"))
    
    # Predicted mask overlay (red)
    pred_mask_rgb = np.zeros_like(original_image)
    pred_mask_rgb[pred_mask > 0] = [255, 0, 0]  # Red for predicted manipulated regions
    
    # Create predicted mask overlay
    pred_overlay = original_image.copy()
    pred_mask_indices = pred_mask > 0
    pred_overlay[pred_mask_indices] = (
        (1-alpha) * original_image[pred_mask_indices] + alpha * pred_mask_rgb[pred_mask_indices]
    ).astype(np.uint8)
    
    # Save predicted mask overlay
    Image.fromarray(pred_overlay).save(os.path.join(output_path, f"{img_name}_pred.png"))
    
    # Save predicted binary mask
    Image.fromarray(pred_mask * 255).convert('L').save(os.path.join(output_path, f"{img_name}_pred_binary.png"))
    
    # If ground truth mask is provided
    if gt_mask is not None:
        # Ground truth mask overlay (green)
        gt_mask_rgb = np.zeros_like(original_image)
        gt_mask_rgb[gt_mask > 0] = [0, 255, 0]  # Green for ground truth
        
        # Create ground truth overlay
        gt_overlay = original_image.copy()
        gt_mask_indices = gt_mask > 0
        gt_overlay[gt_mask_indices] = (
            (1-alpha) * original_image[gt_mask_indices] + alpha * gt_mask_rgb[gt_mask_indices]
        ).astype(np.uint8)
        
        # Save ground truth overlay
        Image.fromarray(gt_overlay).save(os.path.join(output_path, f"{img_name}_gt.png"))
        
        # Save ground truth binary mask
        Image.fromarray(gt_mask * 255).convert('L').save(os.path.join(output_path, f"{img_name}_gt_binary.png"))
        
        # Create combined visualization (red = prediction, green = ground truth)
        combined = original_image.copy()
        # Where both masks agree (true positive)
        both_indices = np.logical_and(pred_mask > 0, gt_mask > 0)
        # Where only prediction mask is positive (false positive)
        only_pred_indices = np.logical_and(pred_mask > 0, gt_mask == 0)
        # Where only ground truth mask is positive (false negative)
        only_gt_indices = np.logical_and(pred_mask == 0, gt_mask > 0)
        
        # Yellow for true positives (where both agree)
        combined[both_indices] = (
            (1-alpha) * original_image[both_indices] + alpha * np.array([255, 255, 0])
        ).astype(np.uint8)
        
        # Red for false positives
        combined[only_pred_indices] = (
            (1-alpha) * original_image[only_pred_indices] + alpha * np.array([255, 0, 0])
        ).astype(np.uint8)
        
        # Green for false negatives
        combined[only_gt_indices] = (
            (1-alpha) * original_image[only_gt_indices] + alpha * np.array([0, 255, 0])
        ).astype(np.uint8)
        
        # Save combined visualization
        Image.fromarray(combined).save(os.path.join(output_path, f"{img_name}_combined.png"))

def load_ground_truth_mask(masks_dir, img_name):
    """Try to load ground truth mask if it exists"""
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_path = os.path.join(masks_dir, img_name + ext)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
            return (mask > 127).astype(np.uint8)
    return None

def main():
    parser = argparse.ArgumentParser(description='Visualize images and masks')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Root directory containing images folder with images to visualize')
    parser.add_argument('--combine_gt', action='store_true',
                      help='If set, also show ground truth masks from masks folder')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--num_images', type=int, default=100,
                      help='Number of random images to visualize')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for binary prediction')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size for inference (model input)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import OpenCV here to avoid potential issues
    import cv2
    
    # Configure model
    config.defrost()
    config.MODEL.NAME = 'detconfcmx'
    config.MODEL.MODS = ('RGB','NP++')
    config.MODEL.PRETRAINED = ''  # Empty to avoid loading pretrained weights
    config.DATASET.NUM_CLASSES = 2
    config.MODEL.EXTRA.BACKBONE = 'mit_b2'
    config.MODEL.EXTRA.DECODER = 'MLPDecoder'
    config.MODEL.EXTRA.DECODER_EMBED_DIM = 512
    config.MODEL.EXTRA.PREPRC = 'imagenet'
    config.MODEL.EXTRA.NP_WEIGHTS = ''  # Empty to avoid loading Noiseprint weights
    config.MODEL.EXTRA.MODULES = ['NP++','backbone','loc_head','conf_head','det_head']
    config.MODEL.EXTRA.FIX_MODULES = []  # Empty to avoid fixing modules
    config.MODEL.EXTRA.DETECTION = 'confpool'
    config.MODEL.EXTRA.CONF = True
    config.MODEL.EXTRA.BN_EPS = 0.001
    config.MODEL.EXTRA.BN_MOMENTUM = 0.1
    config.freeze()
    
    # Create and load model
    print(f"Loading model from {args.model_path}")
    try:
        # Create a new model instance
        model = get_model(config)
        
        # Load the state dict
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # If checkpoint is a dict with 'state_dict' key
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's directly the state dict
            state_dict = checkpoint
        
        # Load the state dict into the model
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Get images from the images subfolder
    images_dir = os.path.join(args.input_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Images directory not found at {images_dir}")
        print("Looking directly in the input directory...")
        images_dir = args.input_dir
    
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    if len(image_files) == 0:
        print(f"No images found in {images_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images")
    
    # Select random subset
    num_images = min(args.num_images, len(image_files))
    selected_images = random.sample(image_files, num_images)
    print(f"Selected {len(selected_images)} random images for visualization")
    
    # Process images in batches
    batch_size = args.batch_size
    img_size = (args.img_size, args.img_size)
    
    for i in range(0, len(selected_images), batch_size):
        # Get batch of image paths
        batch_paths = selected_images[i:i+batch_size]
        batch_names = [os.path.splitext(os.path.basename(path))[0] for path in batch_paths]
        
        # Load batch of images
        batch_tensors, original_images = load_images(batch_paths, img_size)
        
        # Process batch
        pred_np, binary_masks = process_batch(model, batch_tensors, device, args.threshold)
        
        # Check for masks folder for ground truth
        masks_dir = None
        if args.combine_gt:
            masks_dir = os.path.join(args.input_dir, 'masks')
            if not os.path.exists(masks_dir):
                print(f"Warning: Masks directory not found at {masks_dir}")
                masks_dir = None
        
        # Save visualizations
        for j, (img_name, original_img, mask) in enumerate(zip(batch_names, original_images, binary_masks)):
            # Resize mask to match original image if needed
            if mask.shape != original_img.shape[:2]:
                mask_resized = cv2.resize(
                    mask, 
                    (original_img.shape[1], original_img.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_resized = mask
            
            # Load ground truth mask if available
            gt_mask = None
            if masks_dir is not None:
                gt_mask = load_ground_truth_mask(masks_dir, img_name)
                if gt_mask is not None and gt_mask.shape != original_img.shape[:2]:
                    gt_mask = cv2.resize(
                        gt_mask,
                        (original_img.shape[1], original_img.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
            
            # Save visualization
            save_visualization(original_img, mask_resized, args.output_dir, img_name, gt_mask)
            
            print(f"Processed {i+j+1}/{len(selected_images)}: {img_name}")
    
    print(f"Visualizations saved to {args.output_dir}")
    print(f"Done! Processed {len(selected_images)} images.")

if __name__ == '__main__':
    main()