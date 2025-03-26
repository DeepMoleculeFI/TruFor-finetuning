#!/usr/bin/env python
"""
Script to check and verify the dataset organization for TruFor training
"""

import os
import glob
import argparse
from pathlib import Path

def count_files(pattern):
    """Count files matching a glob pattern"""
    return len(glob.glob(pattern, recursive=True))

def main():
    parser = argparse.ArgumentParser(description='Check dataset organization for TruFor')
    parser.add_argument('--data_root', type=str, default='/root/cv_hack_25/data',
                       help='Root directory of the dataset')
    
    args = parser.parse_args()
    data_root = args.data_root
    
    # Check if data root exists
    if not os.path.exists(data_root):
        print(f"ERROR: Data root '{data_root}' does not exist!")
        return
    
    print(f"Checking dataset in: {data_root}")
    
    # Training data
    train_dir = os.path.join(data_root, 'train/train')
    train_images_dir = os.path.join(train_dir, 'images')
    train_masks_dir = os.path.join(train_dir, 'masks')
    train_originals_dir = os.path.join(train_dir, 'originals')
    
    # Test data
    test_dir = os.path.join(data_root, 'test/test')
    
    # Check directories
    print("\nChecking directory structure:")
    for directory in [train_dir, train_images_dir, train_masks_dir, train_originals_dir, test_dir]:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} does not exist")
    
    # Count files
    print("\nCounting files:")
    train_images = count_files(os.path.join(train_images_dir, '*.*'))
    train_masks = count_files(os.path.join(train_masks_dir, '*.*'))
    train_originals = count_files(os.path.join(train_originals_dir, '*.*'))
    test_images = count_files(os.path.join(test_dir, '*.*'))
    
    print(f"Training images: {train_images}")
    print(f"Training masks: {train_masks}")
    print(f"Training originals: {train_originals}")
    print(f"Test images: {test_images}")
    
    # Check file extensions
    print("\nChecking file extensions:")
    train_image_extensions = set([Path(f).suffix for f in glob.glob(os.path.join(train_images_dir, '*.*'))])
    train_mask_extensions = set([Path(f).suffix for f in glob.glob(os.path.join(train_masks_dir, '*.*'))])
    train_original_extensions = set([Path(f).suffix for f in glob.glob(os.path.join(train_originals_dir, '*.*'))])
    test_extensions = set([Path(f).suffix for f in glob.glob(os.path.join(test_dir, '*.*'))])
    
    print(f"Training image extensions: {train_image_extensions}")
    print(f"Training mask extensions: {train_mask_extensions}")
    print(f"Training original extensions: {train_original_extensions}")
    print(f"Test image extensions: {test_extensions}")
    
    # Validate matching files
    print("\nValidating file matching:")
    image_basenames = set([Path(f).stem for f in glob.glob(os.path.join(train_images_dir, '*.*'))])
    mask_basenames = set([Path(f).stem for f in glob.glob(os.path.join(train_masks_dir, '*.*'))])
    original_basenames = set([Path(f).stem for f in glob.glob(os.path.join(train_originals_dir, '*.*'))])
    
    if image_basenames == mask_basenames:
        print("✓ Each image has a matching mask")
    else:
        print("✗ Mismatch between images and masks")
        print(f"  {len(image_basenames - mask_basenames)} images without masks")
        print(f"  {len(mask_basenames - image_basenames)} masks without images")
    
    if image_basenames == original_basenames:
        print("✓ Each image has a matching original")
    else:
        print("✗ Mismatch between images and originals")
        print(f"  {len(image_basenames - original_basenames)} images without originals")
        print(f"  {len(original_basenames - image_basenames)} originals without images")
    
    # Check image sizes (sample)
    try:
        from PIL import Image
        print("\nChecking image sizes (sample):")
        
        # Sample a few images
        sample_images = glob.glob(os.path.join(train_images_dir, '*.*'))[:5]
        
        for img_path in sample_images:
            base_name = Path(img_path).stem
            img = Image.open(img_path)
            mask_path = glob.glob(os.path.join(train_masks_dir, f"{base_name}.*"))[0]
            mask = Image.open(mask_path)
            original_path = glob.glob(os.path.join(train_originals_dir, f"{base_name}.*"))[0]
            original = Image.open(original_path)
            
            print(f"File: {base_name}")
            print(f"  Image: {img.size}")
            print(f"  Mask: {mask.size}")
            print(f"  Original: {original.size}")
    except Exception as e:
        print(f"Error checking image sizes: {e}")
    
    print("\nDataset check complete!")

if __name__ == '__main__':
    main()
