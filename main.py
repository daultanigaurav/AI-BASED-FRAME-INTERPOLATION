#!/usr/bin/env python3
"""
AI-Based Frame Interpolation - Main Application Entry Point

This script provides a command-line interface for the frame interpolation system.
It can be used to train models, run inference, or start the web API.
"""

import argparse
import sys
import os
import torch
from pathlib import Path

# Add the model directory to the path
sys.path.append(str(Path(__file__).parent / 'model'))

def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Frame Interpolation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python main.py train --data-dir data/train --epochs 100
  
  # Run inference on two frames
  python main.py infer --frame1 frame1.jpg --frame2 frame2.jpg --output output.jpg
  
  # Start the web API
  python main.py serve --host 0.0.0.0 --port 8000
  
  # Interpolate a video
  python main.py video --input video.mp4 --output interpolated.mp4 --factor 2
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the frame interpolation model')
    train_parser.add_argument('--data-dir', required=True, help='Directory containing training data')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference on two frames')
    infer_parser.add_argument('--frame1', required=True, help='Path to first frame')
    infer_parser.add_argument('--frame2', required=True, help='Path to second frame')
    infer_parser.add_argument('--output', required=True, help='Output path for interpolated frame')
    infer_parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    infer_parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    
    # Video interpolation command
    video_parser = subparsers.add_parser('video', help='Interpolate frames in a video')
    video_parser.add_argument('--input', required=True, help='Input video path')
    video_parser.add_argument('--output', required=True, help='Output video path')
    video_parser.add_argument('--factor', type=int, default=2, help='Interpolation factor')
    video_parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    video_parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the web API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    # Model info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', default='best_model.pth', help='Path to model file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    try:
        if args.command == 'train':
            from model.train import main as train_main
            # Modify sys.argv to pass arguments to train script
            sys.argv = ['train.py', '--data-dir', args.data_dir, '--epochs', str(args.epochs)]
            train_main()
            
        elif args.command == 'infer':
            from model.inference import FrameInterpolator
            import cv2
            
            # Load model and run inference
            interpolator = FrameInterpolator(args.model, device)
            
            # Read frames
            frame1 = cv2.imread(args.frame1)
            frame2 = cv2.imread(args.frame2)
            
            if frame1 is None or frame2 is None:
                print(f"Error: Could not read input frames")
                return
            
            # Generate interpolated frame
            print("Generating intermediate frame...")
            interpolated = interpolator.interpolate_frames(frame1, frame2)
            
            # Save result
            cv2.imwrite(args.output, interpolated)
            print(f"Interpolated frame saved to: {args.output}")
            
        elif args.command == 'video':
            from model.inference import FrameInterpolator
            
            # Load model and interpolate video
            interpolator = FrameInterpolator(args.model, device)
            
            print(f"Interpolating video: {args.input}")
            print(f"Output: {args.output}")
            print(f"Factor: {args.factor}x")
            
            interpolator.interpolate_video(args.input, args.output, args.factor)
            print("Video interpolation completed!")
            
        elif args.command == 'serve':
            import uvicorn
            from api.app import app
            
            print(f"Starting API server on {args.host}:{args.port}")
            uvicorn.run(
                app, 
                host=args.host, 
                port=args.port, 
                reload=args.reload
            )
            
        elif args.command == 'info':
            if not os.path.exists(args.model):
                print(f"Model file not found: {args.model}")
                return
            
            # Load model and show info
            checkpoint = torch.load(args.model, map_location='cpu')
            print(f"Model: {args.model}")
            print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"Training Loss: {checkpoint.get('train_loss', 'Unknown'):.6f}")
            print(f"Validation Loss: {checkpoint.get('val_loss', 'Unknown'):.6f}")
            
            # Show model architecture info
            from model.unet import FrameInterpolationUNet
            model = FrameInterpolationUNet()
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")
            
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed and the project structure is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
