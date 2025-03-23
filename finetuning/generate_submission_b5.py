#!/usr/bin/env python
"""
Optimized test script for generating predictions using a trained model
Creates submission file in the required format: id,rle
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor
import time

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
from lib.config import config
from lib.utils import get_model

def mask2rle(img):
    """Convert mask to RLE"""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_images(image_paths, img_size=(256, 256)):
    """Load and preprocess a batch of images in parallel"""
    images = []
    
    def load_image(path):
        image = Image.open(path).convert('RGB')
        if img_size:
            image = image.resize(img_size, Image.BILINEAR)
        img_array = np.array(image)
        return img_array
    
    # Use thread pool for parallel loading
    with ThreadPoolExecutor(max_workers=4) as executor:
        images = list(executor.map(load_image, image_paths))
    
    # Stack images into a batch
    batch = np.stack(images)
    # Convert to tensor [B, H, W, C] -> [B, C, H, W]
    batch_tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
    
    return batch_tensor

def process_batch(model, image_batch, device, threshold=0.5):
    """Process a batch of images"""
    with torch.no_grad():
        # Move batch to device
        image_batch = image_batch.to(device)
        
        # Run inference
        outputs, _, _, _ = model(image_batch)
        
        # Get prediction maps (class 1 is "manipulated")
        preds = F.softmax(outputs, dim=1)[:, 1]
        
        # Apply threshold
        pred_np = preds.cpu().numpy()
        binary_masks = (pred_np > threshold).astype(np.uint8)
        
    return binary_masks

def main():
    parser = argparse.ArgumentParser(description='Generate test predictions')
    parser.add_argument('--test_dir', type=str, default='/root/cv_hack_25/data/test/test',
                      help='Directory containing test images')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='submission',
                      help='Directory to save output files')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for binary prediction')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Image size for inference')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configure model
    config.defrost()
    config.MODEL.NAME = 'detconfcmx'
    config.MODEL.MODS = ('RGB','NP++')
    config.MODEL.PRETRAINED = ''  # Empty to avoid loading pretrained weights
    config.DATASET.NUM_CLASSES = 2
    config.MODEL.EXTRA.BACKBONE = 'mit_b5'
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
    
    # Get test images
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_images.extend(glob.glob(os.path.join(args.test_dir, ext)))
    test_images = sorted(test_images)
    print(f"Found {len(test_images)} test images")
    
    # Create submission data list
    submission_data = []
    
    # Process images in batches
    total_time = 0
    batch_size = args.batch_size
    img_size = (args.img_size, args.img_size)
    
    for i in range(0, len(test_images), batch_size):
        start_time = time.time()
        
        # Get batch of image paths
        batch_paths = test_images[i:i+batch_size]
        batch_ids = [os.path.splitext(os.path.basename(path))[0] for path in batch_paths]
        
        # Load batch of images
        batch_tensors = load_images(batch_paths, img_size)
        
        # Process batch
        binary_masks = process_batch(model, batch_tensors, device, args.threshold)
        
        # Convert each mask to RLE and add to submission data
        for j, (image_id, mask) in enumerate(zip(batch_ids, binary_masks)):
            rle = mask2rle(mask)
            submission_data.append({'ImageId': image_id, 'EncodedPixels': rle})
        
        batch_time = time.time() - start_time
        total_time += batch_time
        
        # Print progress
        batch_num = i // batch_size + 1
        total_batches = (len(test_images) + batch_size - 1) // batch_size
        print(f"Processed batch {batch_num}/{total_batches} in {batch_time:.2f}s "
              f"({len(batch_paths)/(batch_time+1e-6):.2f} img/s)")
    
    # Create submission dataframe
    submission_df = pd.DataFrame(submission_data)
    
    # Save submission file
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")
    
    # Print stats
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average processing time per image: {total_time/len(test_images):.4f}s")
    print(f"Processed {len(test_images)/total_time:.2f} images per second")
    
    # Verify the format
    print("First few rows of submission file:")
    try:
        print(pd.read_csv(submission_path).head())
    except Exception as e:
        print(f"Error reading submission file: {e}")

if __name__ == '__main__':
    main()