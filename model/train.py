import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from tqdm import tqdm
import argparse
from pathlib import Path
from unet import FrameInterpolationUNet

# Add the parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

class SSIMLoss(nn.Module):
    """SSIM Loss for image quality assessment"""
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self._create_window(window_size, channel)

    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

class CombinedLoss(nn.Module):
    """Combined MSE and SSIM loss"""
    def __init__(self, mse_weight=0.5, ssim_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        return self.mse_weight * mse + self.ssim_weight * ssim

class FrameTripletDataset(Dataset):
    """Dataset for loading frame triplets (t0, t1, ground_truth_mid)"""
    def __init__(self, data_dir, sequence_length=3):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.triplets = self._load_triplets()
        
    def _load_triplets(self):
        triplets = []
        
        # Walk through the data directory
        for video_dir in os.listdir(self.data_dir):
            video_path = os.path.join(self.data_dir, video_dir)
            if os.path.isdir(video_path):
                # Get all frame files
                frame_files = sorted([f for f in os.listdir(video_path) 
                                   if f.endswith(('.jpg', '.png', '.bmp'))])
                
                # Create triplets: (frame_t0, frame_t1, ground_truth_mid)
                # We'll use frame_0, frame_2 as input and frame_1 as target
                for i in range(len(frame_files) - 2):
                    triplet = {
                        'video_dir': video_path,
                        'frame_t0': frame_files[i],      # First frame
                        'frame_t1': frame_files[i + 2],  # Third frame  
                        'ground_truth': frame_files[i + 1]  # Middle frame (target)
                    }
                    triplets.append(triplet)
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # Load frames
        frame_t0_path = os.path.join(triplet['video_dir'], triplet['frame_t0'])
        frame_t1_path = os.path.join(triplet['video_dir'], triplet['frame_t1'])
        ground_truth_path = os.path.join(triplet['video_dir'], triplet['ground_truth'])
        
        # Read frames as grayscale
        frame_t0 = cv2.imread(frame_t0_path, cv2.IMREAD_GRAYSCALE)
        frame_t1 = cv2.imread(frame_t1_path, cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to standard size (256x256)
        frame_t0 = cv2.resize(frame_t0, (256, 256))
        frame_t1 = cv2.resize(frame_t1, (256, 256))
        ground_truth = cv2.resize(ground_truth, (256, 256))
        
        # Normalize to [0, 1] and convert to tensor
        frame_t0 = frame_t0.astype(np.float32) / 255.0
        frame_t1 = frame_t1.astype(np.float32) / 255.0
        ground_truth = ground_truth.astype(np.float32) / 255.0
        
        # Convert to tensors and add channel dimension
        frame_t0 = torch.from_numpy(frame_t0).unsqueeze(0)  # [1, H, W]
        frame_t1 = torch.from_numpy(frame_t1).unsqueeze(0)  # [1, H, W]
        ground_truth = torch.from_numpy(ground_truth).unsqueeze(0)  # [1, H, W]
        
        return frame_t0, frame_t1, ground_truth

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """Train the model with combined MSE and SSIM loss"""
    
    # Loss function
    criterion = CombinedLoss(mse_weight=0.5, ssim_weight=0.5)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Using device: {device}")
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total validation samples: {len(val_loader.dataset)}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (frame_t0, frame_t1, ground_truth) in enumerate(train_bar):
            frame_t0, frame_t1, ground_truth = frame_t0.to(device), frame_t1.to(device), ground_truth.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(frame_t0, frame_t1)
            
            # Calculate loss
            loss = criterion(output, ground_truth)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for frame_t0, frame_t1, ground_truth in val_bar:
                frame_t0, frame_t1, ground_truth = frame_t0.to(device), frame_t1.to(device), ground_truth.to(device)
                
                output = model(frame_t0, frame_t1)
                loss = criterion(output, ground_truth)
                val_loss += loss.item()
                
                val_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }
            torch.save(checkpoint, 'best_model.pth')
            print(f'  New best model saved! (Val Loss: {best_val_loss:.6f})')
        
        print('-' * 50)
    
    print(f"Training completed! Best validation loss: {best_val_loss:.6f}")
    return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(description='Train Frame Interpolation UNet')
    parser.add_argument('--data-dir', required=True, help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create datasets
    full_dataset = FrameTripletDataset(args.data_dir)
    
    # Split into train and validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    model = FrameInterpolationUNet(bilinear=True)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, device=device
    )
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
