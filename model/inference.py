import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
from unet import FrameInterpolationUNet

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess image: resize to target size and normalize to [-1, 1]
    
    Args:
        image_path: Path to input image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed tensor of shape [1, 1, H, W] with values in [-1, 1]
    """
    # Read image as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Resize to target size
    image = cv2.resize(image, target_size)
    
    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Normalize to [-1, 1]
    image = 2.0 * image - 1.0
    
    # Convert to tensor and add batch and channel dimensions
    # Shape: [1, 1, H, W]
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    
    return image_tensor

def postprocess_image(tensor):
    """
    Postprocess tensor: convert from [-1, 1] to [0, 255] uint8
    
    Args:
        tensor: Input tensor with values in [-1, 1]
    
    Returns:
        Postprocessed image as uint8 array with values in [0, 255]
    """
    # Convert from [-1, 1] to [0, 1]
    image = (tensor + 1.0) / 2.0
    
    # Clip values to [0, 1]
    image = torch.clamp(image, 0.0, 1.0)
    
    # Convert to numpy and scale to [0, 255]
    image_np = image.squeeze().cpu().numpy()
    image_uint8 = (image_np * 255).astype(np.uint8)
    
    return image_uint8

def load_model(model_path, device):
    """
    Load trained U-Net model
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Create model
    model = FrameInterpolationUNet(bilinear=True)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        print(f"Trained for {checkpoint.get('epoch', 'Unknown')} epochs")
        print(f"Best validation loss: {checkpoint.get('val_loss', 'Unknown'):.6f}")
    else:
        # If it's just the state dict
        model.load_state_dict(checkpoint)
        print(f"Model state dict loaded from {model_path}")
    
    model = model.to(device)
    model.eval()
    
    return model

def interpolate_frames(model, frame1, frame2, device):
    """
    Generate interpolated frame between two input frames
    
    Args:
        model: Trained U-Net model
        frame1: First frame tensor [1, 1, H, W]
        frame2: Second frame tensor [1, 1, H, W]
        device: Device to run inference on
    
    Returns:
        Interpolated frame tensor
    """
    # Move inputs to device
    frame1 = frame1.to(device)
    frame2 = frame2.to(device)
    
    # Generate interpolated frame
    with torch.no_grad():
        interpolated = model(frame1, frame2)
    
    return interpolated

def main():
    parser = argparse.ArgumentParser(description='Frame Interpolation Inference')
    parser.add_argument('--frame1', required=True, help='Path to first input frame')
    parser.add_argument('--frame2', required=True, help='Path to second input frame')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--output', default='output.png', help='Output image path')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # Load and preprocess input frames
        print("Loading and preprocessing input frames...")
        frame1 = preprocess_image(args.frame1)
        frame2 = preprocess_image(args.frame2)
        
        print(f"Frame 1 shape: {frame1.shape}")
        print(f"Frame 2 shape: {frame2.shape}")
        
        # Load trained model
        print("Loading trained model...")
        model = load_model(args.model, device)
        
        # Generate interpolated frame
        print("Generating interpolated frame...")
        interpolated = interpolate_frames(model, frame1, frame2, device)
        
        print(f"Interpolated frame shape: {interpolated.shape}")
        
        # Postprocess and save result
        print("Postprocessing and saving result...")
        output_image = postprocess_image(interpolated)
        
        # Save result
        cv2.imwrite(args.output, output_image)
        print(f"Interpolated frame saved to: {args.output}")
        
        # Print some statistics
        print(f"Input frame 1 range: [{frame1.min().item():.3f}, {frame1.max().item():.3f}]")
        print(f"Input frame 2 range: [{frame2.min().item():.3f}, {frame2.max().item():.3f}]")
        print(f"Output range: [{interpolated.min().item():.3f}, {interpolated.max().item():.3f}]")
        print(f"Output image range: [{output_image.min()}, {output_image.max()}]")
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
