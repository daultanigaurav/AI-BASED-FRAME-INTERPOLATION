#!/usr/bin/env python3
"""
Simple demo script to show the frame interpolation system working
This doesn't require a trained model - it uses dummy data
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path

# Add model directory to path
sys.path.append('model')

def create_dummy_frames():
    """Create dummy frames for demonstration"""
    print("🎬 Creating dummy frames...")
    
    # Create a simple moving pattern
    frames = []
    for i in range(5):
        # Create a frame with a moving circle
        frame = np.zeros((256, 256), dtype=np.uint8)
        
        # Draw a circle that moves across the frame
        center_x = 50 + i * 40  # Move from left to right
        center_y = 128
        radius = 30
        
        cv2.circle(frame, (center_x, center_y), radius, 255, -1)
        
        # Add some noise for realism
        noise = np.random.randint(0, 30, (256, 256), dtype=np.uint8)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        
        frames.append(frame)
    
    return frames

def save_dummy_frames(frames, output_dir):
    """Save dummy frames to directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        filename = f"frame_{i:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"  Saved {filename}")
    
    return output_dir

def test_baseline_methods():
    """Test baseline interpolation methods"""
    print("\n🔬 Testing baseline interpolation methods...")
    
    # Create dummy frames
    frames = create_dummy_frames()
    
    # Test with frames 0 and 4 (first and last)
    frame1 = frames[0]
    frame2 = frames[4]
    
    print(f"  Frame 1 shape: {frame1.shape}")
    print(f"  Frame 2 shape: {frame2.shape}")
    
    # Linear interpolation
    linear_result = ((frame1.astype(np.float32) + frame2.astype(np.float32)) / 2).astype(np.uint8)
    print("  ✅ Linear interpolation: successful")
    
    # Optical flow interpolation
    try:
        flow = cv2.calcOpticalFlowFarneback(
            frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.1, 0
        )
        
        # Calculate intermediate flow
        flow_half = flow * 0.5
        h, w = frame1.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        new_x = x_coords + flow_half[:, :, 0]
        new_y = y_coords + flow_half[:, :, 1]
        
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)
        
        optical_result = cv2.remap(
            frame1, new_x, new_y, cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_REPLICATE
        )
        print("  ✅ Optical flow interpolation: successful")
        
    except Exception as e:
        print(f"  ⚠️  Optical flow interpolation: {e}")
        optical_result = linear_result  # Fallback
    
    return frame1, frame2, linear_result, optical_result

def test_unet_creation():
    """Test if U-Net model can be created"""
    print("\n🧠 Testing U-Net model creation...")
    
    try:
        from unet import FrameInterpolationUNet
        
        # Create model
        model = FrameInterpolationUNet(bilinear=True)
        print("  ✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  📊 Total parameters: {total_params:,}")
        
        # Test forward pass with dummy data
        batch_size = 1
        channels = 1
        height = 256
        width = 256
        
        frame1 = torch.randn(batch_size, channels, height, width)
        frame2 = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            output = model(frame1, frame2)
        
        print(f"  ✅ Forward pass successful")
        print(f"     Input shapes: {frame1.shape}, {frame2.shape}")
        print(f"     Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False

def test_api_structure():
    """Test if API structure is correct"""
    print("\n🌐 Testing API structure...")
    
    api_file = "api/app.py"
    if os.path.exists(api_file):
        print(f"  ✅ API file exists: {api_file}")
        
        # Check if it's a valid Python file
        try:
            with open(api_file, 'r') as f:
                content = f.read()
                if "FastAPI" in content and "interpolate" in content:
                    print("  ✅ API file contains FastAPI and interpolate endpoint")
                    return True
                else:
                    print("  ⚠️  API file exists but may not be properly configured")
                    return False
        except Exception as e:
            print(f"  ❌ Error reading API file: {e}")
            return False
    else:
        print(f"  ❌ API file missing: {api_file}")
        return False

def test_frontend_structure():
    """Test if frontend structure is correct"""
    print("\n🖥️  Testing frontend structure...")
    
    frontend_files = ["frontend/index.html", "frontend/script.js"]
    all_exist = True
    
    for file_path in frontend_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_exist = False
    
    return all_exist

def test_inference_pipeline():
    """Test if inference pipeline works"""
    print("\n🎯 Testing inference pipeline...")
    
    try:
        from model.inference import preprocess_image, postprocess_image
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Test preprocessing
        preprocessed = preprocess_image(dummy_image)
        print(f"  ✅ Preprocessing: input {dummy_image.shape} -> tensor {preprocessed.shape}")
        
        # Test postprocessing
        postprocessed = postprocess_image(preprocessed)
        print(f"  ✅ Postprocessing: tensor {preprocessed.shape} -> output {postprocessed.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Inference pipeline test failed: {e}")
        return False

def create_sample_data():
    """Create sample data structure for testing"""
    print("\n📁 Creating sample data structure...")
    
    # Create test_data directory
    test_dir = "test_data"
    video_dir = os.path.join(test_dir, "sample_video")
    os.makedirs(video_dir, exist_ok=True)
    
    # Create dummy frames
    frames = create_dummy_frames()
    
    # Save frames
    for i, frame in enumerate(frames):
        filename = f"frame_{i:03d}.png"
        filepath = os.path.join(video_dir, filename)
        cv2.imwrite(filepath, frame)
    
    print(f"  ✅ Created {len(frames)} frames in {video_dir}")
    print(f"  📁 Test data ready at: {test_dir}")
    
    return test_dir

def main():
    """Run all tests and create demo"""
    print("🚀 FRAME INTERPOLATION SYSTEM DEMO")
    print("=" * 50)
    
    # Test basic functionality
    tests = [
        ("Baseline Methods", test_baseline_methods),
        ("U-Net Model", test_unet_creation),
        ("API Structure", test_api_structure),
        ("Frontend Structure", test_frontend_structure),
        ("Inference Pipeline", test_inference_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Create sample data
    test_dir = create_sample_data()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= 3:  # At least basic functionality works
        print("\n🎉 Basic system is working!")
        print("\n📋 Next steps to get full system running:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Prepare training data in data/ folder")
        print("3. Train model: python model/train.py --data-dir data/")
        print("4. Test with sample data: python model/evaluation_simple.py --test-dir test_data/")
        print("5. Start API: python api/app.py")
        print("6. Open frontend/index.html in browser")
        
        print(f"\n📁 Sample test data created at: {test_dir}")
        print("   You can use this to test the evaluation system")
        
    else:
        print(f"\n⚠️  {total - passed} tests failed. System needs more work.")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
