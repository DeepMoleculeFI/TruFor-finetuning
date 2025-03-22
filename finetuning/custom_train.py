
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Add the TruFor path to the system path
trufor_path = "/root/cv_hack_25/TruFor/TruFor_train_test"
sys.path.insert(0, trufor_path)

# Import required modules
from lib.config import config, update_config
from lib.utils import get_model, create_logger, FullModel

# Import custom dataset
from ai_manip_dataset import AIManipDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train TruFor for AI Manipulation Detection')
    parser.add_argument('--data_root', type=str, default='/root/cv_hack_25/data', 
                      help='Root directory for dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
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
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs, _, _, _ = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}')
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs, _, _, _ = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}')
        
        # Save model
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f'{args.output_dir}/model_epoch{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), f'{args.output_dir}/model_final.pth')
    print('Training complete!')

if __name__ == '__main__':
    main()
