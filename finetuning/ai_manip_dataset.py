
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class AIManipDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        
        # Define paths based on your directory structure
        if split == 'train' or split == 'valid':
            self.data_dir = os.path.join(root_dir, 'train', 'train')
        else:
            self.data_dir = os.path.join(root_dir, 'test', 'test')
        
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.masks_dir = os.path.join(self.data_dir, 'masks')
        
        # Get list of image files
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_files.extend(glob.glob(os.path.join(self.images_dir, ext)))
        
        if split == 'valid':
            # Use 10% of training data for validation
            np.random.seed(42)
            indices = np.random.permutation(len(self.image_files))
            valid_size = int(0.1 * len(self.image_files))
            self.image_files = [self.image_files[i] for i in indices[:valid_size]]
        elif split == 'train':
            # Use 90% of training data for training
            np.random.seed(42)
            indices = np.random.permutation(len(self.image_files))
            valid_size = int(0.1 * len(self.image_files))
            self.image_files = [self.image_files[i] for i in indices[valid_size:]]
        
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
