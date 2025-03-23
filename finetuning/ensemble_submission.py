#!/usr/bin/env python
"""
Ensemble test script for generating predictions using two trained TruFor models 
Creates submission file in the required format: ImageId,EncodedPixels
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

# Fix temporary directory issue before importing any modules that might need it
os.environ['TMPDIR'] = os.path.expanduser('~/temp')
os.makedirs(os.environ['TMPDIR'], exist_ok=True)
os.environ['WANDB_MODE'] = 'disabled'  # Disable wandb to avoid related issues

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
import torch
import torch.nn.functional as F

# Import TruFor modules
# We'll load these after fixing the temp directory and configuring the environment
from lib.config import config
from lib.utils import get_model

def mask2rle(img):
    """Convert binary mask to RLE."""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_images(image_paths, img_size=(256, 256)):
    """Load and preprocess a batch of images in parallel."""
    images = []
    
    def load_image(path):
        try:
            image = Image.open(path).convert('RGB')
            if img_size:
                image = image.resize(img_size, Image.BILINEAR)
            img_array = np.array(image)
            return img_array
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a black image as fallback
            if img_size:
                return np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            else:
                return np.zeros((256, 256, 3), dtype=np.uint8)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        images = list(executor.map(load_image, image_paths))
    
    batch = np.stack(images)
    # Convert to tensor [B, H, W, C] -> [B, C, H, W]
    batch_tensor = torch.from_numpy(batch).float().permute(0, 3, 1, 2) / 255.0
    
    return batch_tensor

def configure_model(backbone):
    """
    Configure TruFor model with specified backbone
    """
    config.defrost()
    config.MODEL.NAME = 'detconfcmx'
    config.MODEL.MODS = ('RGB','NP++')
    config.MODEL.PRETRAINED = ''  
    config.DATASET.NUM_CLASSES = 2
    
    # Set backbone
    config.MODEL.EXTRA.BACKBONE = backbone
    config.MODEL.EXTRA.DECODER = 'MLPDecoder'
    config.MODEL.EXTRA.DECODER_EMBED_DIM = 512
    config.MODEL.EXTRA.PREPRC = 'imagenet'
    config.MODEL.EXTRA.NP_WEIGHTS = ''  
    config.MODEL.EXTRA.MODULES = ['NP++','backbone','loc_head','conf_head','det_head']
    config.MODEL.EXTRA.FIX_MODULES = []
    config.MODEL.EXTRA.DETECTION = 'confpool'
    config.MODEL.EXTRA.BN_EPS = 0.001
    config.MODEL.EXTRA.BN_MOMENTUM = 0.1
    config.freeze()
    
    return config

def load_trufor_model(backbone, checkpoint_path, device):
    """
    Create a TruFor model for the specified backbone, 
    then load the given checkpoint.
    """
    # Configure model
    configure_model(backbone)
    
    # Create model
    try:
        model = get_model(config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"Successfully loaded {backbone} model")
        return model
    except Exception as e:
        print(f"Error loading {backbone} model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_batch_ensemble(models, image_batch, device, threshold=0.5):
    """
    Forward the batch through all models, average their predicted class-1
    probabilities, then threshold to produce a single ensemble mask.
    """
    with torch.no_grad():
        image_batch = image_batch.to(device)
        
        # Accumulate predictions from each model
        ensemble_preds = None
        model_count = len(models)
        weights = [1.0/model_count] * model_count  # Equal weighting
        
        for i, model in enumerate(models):
            try:
                outputs, _, _, _ = model(image_batch)
                # Probability of class 1 (manipulated)
                probs = F.softmax(outputs, dim=1)[:, 1]
                if ensemble_preds is None:
                    ensemble_preds = probs * weights[i]
                else:
                    ensemble_preds += probs * weights[i]
            except Exception as e:
                print(f"Error during inference with model {i}: {e}")
                # If a model fails, adjust weights of remaining models
                if ensemble_preds is None and i < len(models) - 1:
                    # First model failed, adjust weights for remaining models
                    total_weight = sum(weights[i+1:])
                    if total_weight > 0:
                        weights[i+1:] = [w/total_weight for w in weights[i+1:]]
                    continue
                elif ensemble_preds is not None:
                    # Not the first model, continue with what we have
                    continue
                else:
                    # All models failed
                    raise RuntimeError("All models failed during inference")
        
        # Threshold
        pred_np = ensemble_preds.cpu().numpy()
        binary_masks = (pred_np > threshold).astype(np.uint8)
        
    return binary_masks

def main():
    parser = argparse.ArgumentParser(description='Generate ensemble test predictions')
    parser.add_argument('--test_dir', type=str, default='/root/cv_hack_25/data/test/test',
                        help='Directory containing test images')
    parser.add_argument('--b2_model_path', type=str, required=True,
                        help='Path to trained mit_b2 model')
    parser.add_argument('--b5_model_path', type=str,
                        help='Path to trained mit_b5 model (optional)')
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
    
    # Load models
    models = []
    
    # Always load the B2 model
    print(f"Loading mit_b2 from {args.b2_model_path}")
    model_b2 = load_trufor_model('mit_b2', args.b2_model_path, device)
    models.append(model_b2)
    
    # Optionally load the B5 model if path is provided
    if args.b5_model_path:
        print(f"Loading mit_b5 from {args.b5_model_path}")
        model_b5 = load_trufor_model('mit_b5', args.b5_model_path, device)
        models.append(model_b5)
    
    # Gather test images
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_images.extend(glob.glob(os.path.join(args.test_dir, ext)))
    test_images = sorted(test_images)
    print(f"Found {len(test_images)} test images")
    
    # Submission data container
    submission_data = []
    
    total_time = 0
    batch_size = args.batch_size
    img_size = (args.img_size, args.img_size)
    
    for i in range(0, len(test_images), batch_size):
        start_time = time.time()
        
        # Batch of image paths
        batch_paths = test_images[i : i + batch_size]
        batch_ids = [os.path.splitext(os.path.basename(path))[0] for path in batch_paths]
        
        # Load images
        batch_tensors = load_images(batch_paths, img_size)
        
        # Ensemble inference
        binary_masks = process_batch_ensemble(models, batch_tensors, device, args.threshold)
        
        # Convert mask to RLE, store in submission
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
    
    # Convert results to DataFrame and save CSV
    submission_df = pd.DataFrame(submission_data)
    submission_path = os.path.join(args.output_dir, 'ensemble_submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Saved ensemble submission to {submission_path}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average processing time per image: {total_time/len(test_images):.4f}s")
    print(f"Processed {len(test_images)/total_time:.2f} images per second")
    
    # Show first few rows
    print("First few rows of the ensemble submission:")
    try:
        print(pd.read_csv(submission_path).head())
    except Exception as e:
        print(f"Error reading submission file: {e}")

if __name__ == '__main__':
    main()