#!/usr/bin/env python3
"""
Basic test script to check if the frame interpolation system works
"""

import os
import sys
import torch
import numpy as np
import cv2

# Add model directory to path
sys.path.append('model')

def test_basic_imports():
    """Test if basic imports work"""
    print("🔍 Testing basic imports...")
    
    try:
        from unet import FrameInterpolationUNet
        print("✅ U-Net import successful")
    except ImportError as e:
        print(f"❌ U-Net import failed: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if model can be created"""
    print("\n🔍 Testing model creation...")
    
    try:
        from unet import FrameInterpolationUNet
        
        # Create model
        model = FrameInterpolationUNet(bilinear=True)
        print("✅ Model created successfully")
        
        # Test forward pass with dummy data
        batch_size = 1
        channels = 1
        height = 256
        width = 256
        
        # Create dummy input frames
        frame1 = torch.randn(batch_size, channels, height, width)
        frame2 = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        with torch.no_grad():
            output = model(frame1, frame2)
        
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {frame1.shape}, {frame2.shape}")
        print(f"   Output shape: {output.shape}")
        
        # Check output shape
        expected_shape = (batch_size, channels, height, width)
        if output.shape == expected_shape:
            print("✅ Output shape correct")
        else:
            print(f"❌ Output shape incorrect. Expected {expected_shape}, got {output.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_baseline_methods():
    """Test if baseline methods work"""
    print("\n🔍 Testing baseline methods...")
    
    try:
        # Create dummy frames
        frame1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        # Test linear interpolation
        linear_result = (frame1.astype(np.float32) + frame2.astype(np.float32)) / 2
        print("✅ Linear interpolation works")
        
        # Test optical flow (basic)
        try:
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.1, 0
            )
            print("✅ Optical flow calculation works")
        except Exception as e:
            print(f"⚠️  Optical flow failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline methods test failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\n🔍 Testing file structure...")
    
    required_files = [
        'model/unet.py',
        'model/train.py', 
        'model/inference.py',
        'model/evaluation.py',
        'api/app.py',
        'frontend/index.html',
        'frontend/script.js',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files")
        return False
    else:
        print("\n✅ All required files present")
        return True

def test_dependencies():
    """Test if required packages are available"""
    print("\n🔍 Testing dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'cv2', 'numpy', 
        'fastapi', 'uvicorn', 'imageio', 'skimage'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ {package} (OpenCV)")
            elif package == 'skimage':
                import skimage
                print(f"✅ {package}")
            else:
                __import__(package)
                print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages available")
        return True

def main():
    """Run all tests"""
    print("🚀 Starting Basic System Test")
    print("=" * 50)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies),
        ("Basic Imports", test_basic_imports),
        ("Model Creation", test_model_creation),
        ("Baseline Methods", test_baseline_methods)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System should work.")
        print("\nNext steps:")
        print("1. Prepare test data in test_data/ folder")
        print("2. Train model: python model/train.py --data-dir data/")
        print("3. Run evaluation: python model/evaluation.py --test-dir test_data/")
        print("4. Start API: python api/app.py")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Fix issues before proceeding.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
