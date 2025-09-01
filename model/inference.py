import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import imageio
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

def generate_multiple_intermediate_frames(model, frame1, frame2, num_intermediate, device):
    """
    Generate multiple intermediate frames between two input frames
    
    Args:
        model: Trained U-Net model
        frame1: First frame tensor [1, 1, H, W]
        frame2: Second frame tensor [1, 1, H, W]
        num_intermediate: Number of intermediate frames to generate
        device: Device to run inference on
    
    Returns:
        List of intermediate frame tensors
    """
    intermediate_frames = []
    
    # Generate intermediate frames
    for i in range(1, num_intermediate + 1):
        # For now, we'll use the same model to generate intermediate frames
        # In a more sophisticated approach, you could use different interpolation factors
        intermediate = interpolate_frames(model, frame1, frame2, device)
        intermediate_frames.append(intermediate)
        
        print(f"Generated intermediate frame {i}/{num_intermediate}")
    
    return intermediate_frames

def create_smooth_transition_frames(frame1, frame2, num_intermediate):
    """
    Create smooth transition frames using linear interpolation
    This provides a baseline comparison with the AI-generated frames
    
    Args:
        frame1: First frame tensor [1, 1, H, W]
        frame2: Second frame tensor [1, 1, H, W]
        num_intermediate: Number of intermediate frames to generate
    
    Returns:
        List of interpolated frame tensors
    """
    transition_frames = []
    
    for i in range(1, num_intermediate + 1):
        # Linear interpolation factor
        alpha = i / (num_intermediate + 1)
        
        # Linear interpolation between frames
        interpolated = (1 - alpha) * frame1 + alpha * frame2
        transition_frames.append(interpolated)
    
    return transition_frames

def save_frames_as_video(frames, output_path, fps=30):
    """
    Save a list of frames as a video using imageio
    
    Args:
        frames: List of frame arrays (numpy arrays)
        output_path: Path to save the video
        fps: Frames per second for the output video
    """
    print(f"Saving video to {output_path} with {fps} FPS...")
    
    # Convert frames to uint8 if they aren't already
    video_frames = []
    for i, frame in enumerate(frames):
        if frame.dtype != np.uint8:
            # Normalize to [0, 255] if needed
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        video_frames.append(frame)
        print(f"Processed frame {i+1}/{len(frames)}")
    
    # Save as video
    imageio.mimsave(output_path, video_frames, fps=fps)
    print(f"Video saved successfully to {output_path}")
    print(f"Video contains {len(video_frames)} frames at {fps} FPS")

def main():
    parser = argparse.ArgumentParser(description='Frame Interpolation Inference')
    parser.add_argument('--frame1', required=True, help='Path to first input frame')
    parser.add_argument('--frame2', required=True, help='Path to second input frame')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--output', default='output.png', help='Output image path')
    parser.add_argument('--num-intermediate', type=int, default=1, help='Number of intermediate frames to generate')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video')
    parser.add_argument('--save-comparison', action='store_true', help='Save comparison video with linear interpolation')
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
        
        # Generate interpolated frames
        print(f"Generating {args.num_intermediate} intermediate frame(s)...")
        
        if args.num_intermediate == 1:
            # Single frame interpolation
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
            
        else:
            # Multiple frame interpolation
            intermediate_frames = generate_multiple_intermediate_frames(
                model, frame1, frame2, args.num_intermediate, device
            )
            
            print(f"Generated {len(intermediate_frames)} intermediate frames")
            
            # Postprocess all frames
            print("Postprocessing frames...")
            processed_frames = []
            
            # Add first frame
            frame1_processed = postprocess_image(frame1)
            processed_frames.append(frame1_processed)
            
            # Add intermediate frames
            for i, frame in enumerate(intermediate_frames):
                processed_frame = postprocess_image(frame)
                processed_frames.append(processed_frame)
                
                # Save individual intermediate frame
                intermediate_filename = f"intermediate_{i+1:02d}.png"
                cv2.imwrite(intermediate_filename, processed_frame)
                print(f"Saved intermediate frame {i+1} to {intermediate_filename}")
            
            # Add second frame
            frame2_processed = postprocess_image(frame2)
            processed_frames.append(frame2_processed)
            
            # Save video
            video_output = args.output.replace('.png', '.mp4')
            if video_output == args.output:  # If no extension change
                video_output = 'video.mp4'
            
            save_frames_as_video(processed_frames, video_output, args.fps)
            
            # Save comparison video if requested
            if args.save_comparison:
                print("\nGenerating comparison video with linear interpolation...")
                
                # Create linear interpolation frames
                linear_frames = create_smooth_transition_frames(frame1, frame2, args.num_intermediate)
                
                # Process linear frames
                linear_processed_frames = []
                linear_processed_frames.append(frame1_processed)  # First frame
                
                for i, frame in enumerate(linear_frames):
                    processed_frame = postprocess_image(frame)
                    linear_processed_frames.append(processed_frame)
                    
                    # Save individual linear frame
                    linear_filename = f"linear_intermediate_{i+1:02d}.png"
                    cv2.imwrite(linear_filename, processed_frame)
                    print(f"Saved linear frame {i+1} to {linear_filename}")
                
                linear_processed_frames.append(frame2_processed)  # Last frame
                
                # Save comparison video
                comparison_output = video_output.replace('.mp4', '_comparison.mp4')
                save_frames_as_video(linear_processed_frames, comparison_output, args.fps)
                
                print(f"Comparison video saved to: {comparison_output}")
            
            # Print statistics
            print(f"\nAI-generated video: {len(processed_frames)} frames at {args.fps} FPS")
            print(f"Video saved to: {video_output}")
            if args.save_comparison:
                print(f"Comparison video: {len(linear_processed_frames)} frames at {args.fps} FPS")
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
