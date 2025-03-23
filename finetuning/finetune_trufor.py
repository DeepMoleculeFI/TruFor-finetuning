#!/usr/bin/env python
"""
Inference script for AI Manipulation Detection using TruFor
Generates prediction masks and optionally visualization overlays
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
from lib.config import config
from lib.utils import get_model

def create_overlay(image, mask, threshold=0.5, alpha=0.5, color=[0, 0, 255]):
    """
    Create a visualization overlay of the predicted mask on the image
    
    Args:
        image: Original image (numpy array, H x W x 3)
        mask: Prediction mask (numpy array, H x W)
        threshold: Threshold for binary prediction
        alpha: Transparency of the overlay
        color: Color of the overlay (BGR)
        
    Returns:
        Visualization image (numpy array, H x W x 3)
    """
    # Convert image to BGR if it's not already
    if image.shape[2] == 3:
        vis_image = image.copy()
    else:
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Create binary mask
    binary_mask = (mask > threshold).astype(np.uint8)
    
    # Create colored mask
    colored_mask = np.zeros_like(vis_image)
    colored_mask[binary_mask > 0] = color
    
    # Create overlay
    overlay = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
    
    # Add contours for better visibility
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    
    return overlay

def predict_image(model, image_path, device, threshold=0.5):
    """
    Predict manipulation mask for a single image
    
    Args:
        model: TruFor model
        image_path: Path to the image
        device: Device to run inference on
        threshold: Threshold for binary prediction
        
    Returns:
        Tuple of (original image, prediction mask, prediction probability)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Prepare tensor
    image_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Set model to eval mode
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        outputs, conf, det, _ = model(image_tensor)
        
        # Get probability map (class 1 - manipulated)
        prob_map = F.softmax(outputs, dim=1)[0, 1].cpu().numpy()
        
        # Create binary mask
        binary_mask = (prob_map > threshold).astype(np.uint8)
    
    return image_np, binary_mask, prob_map

def process_directory(model, input_dir, output_dir, device, threshold=0.5, create_visualizations=True):
    """
    Process all images in a directory
    
    Args:
        model: TruFor model
        input_dir: Directory containing images
        output_dir: Directory to save outputs
        device: Device to run inference on
        threshold: Threshold for binary prediction
        create_visualizations: Whether to create visualization overlays
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, 'masks')
    os.makedirs(mask_dir, exist_ok=True)
    
    if create_visualizations:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Get list of images
    image_files = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_files.extend(glob.glob(os.path.join(input_dir, f'*.{ext}')))
        image_files.extend(glob.glob(os.path.join(input_dir, f'*.{ext.upper()}')))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        # Get filename without extension
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # Predict mask
        image, binary_mask, prob_map = predict_image(model, image_path, device, threshold)
        
        # Save mask
        mask_path = os.path.join(mask_dir, f"{name}.png")
        cv2.imwrite(mask_path, binary_mask * 255)
        
        # Create and save visualization if requested
        if create_visualizations:
            # Create overlay
            overlay = create_overlay(image, prob_map, threshold)
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f"{name}_overlay.png")
            cv2.imwrite(vis_path, overlay)
            
            # Also save heatmap visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.imshow(prob_map, alpha=0.5, cmap='jet')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{name}_heatmap.png"))
            plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on images using TruFor')
    parser.add_argument('--input', type=str, required=True, 
                      help='Input directory or image file')
    parser.add_argument('--output', type=str, default='output_inference', 
                      help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, 
                      help='GPU ID')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Threshold for binary prediction')
    parser.add_argument('--np_weights', type=str, 
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/noiseprint++/noiseprint++.th',
                      help='Path to Noiseprint++ weights')
    parser.add_argument('--pretrained', type=str,
                      default='/root/cv_hack_25/TruFor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth',
                      help='Path to pretrained backbone weights')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model for inference')
    parser.add_argument('--no_vis', action='store_true',
                      help='Disable visualization generation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input {args.input} does not exist")
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
    
    print(f"Running inference on {args.input}")
    print(f"Using model: {args.model_path}")
    print(f"Noiseprint++ weights: {args.np_weights}")
    print(f"Pretrained weights: {args.pretrained}")
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
            print(f"Model loaded successfully!")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process input
    if os.path.isdir(args.input):
        # Process directory
        process_directory(model, args.input, args.output, device, args.threshold, not args.no_vis)
    elif os.path.isfile(args.input):
        # Process single image
        filename = os.path.basename(args.input)
        name, ext = os.path.splitext(filename)
        
        # Create output directories
        mask_dir = os.path.join(args.output, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        
        if not args.no_vis:
            vis_dir = os.path.join(args.output, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
        
        # Predict mask
        image, binary_mask, prob_map = predict_image(model, args.input, device, args.threshold)
        
        # Save mask
        mask_path = os.path.join(mask_dir, f"{name}.png")
        cv2.imwrite(mask_path, binary_mask * 255)
        
        # Create and save visualization if requested
        if not args.no_vis:
            # Create overlay
            overlay = create_overlay(image, prob_map, args.threshold)
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f"{name}_overlay.png")
            cv2.imwrite(vis_path, overlay)
            
            # Also save heatmap visualization
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.imshow(prob_map, alpha=0.5, cmap='jet')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{name}_heatmap.png"))
            plt.close()
            
        print(f"Processed {args.input}")
        print(f"Mask saved to {mask_path}")
    else:
        print(f"ERROR: Input {args.input} is neither a file nor a directory")
        sys.exit(1)
    
    print("Inference complete!")

if __name__ == '__main__':
    main()