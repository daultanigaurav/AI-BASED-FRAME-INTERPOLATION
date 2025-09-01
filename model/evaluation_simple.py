#!/usr/bin/env python3
"""
Simplified evaluation script for frame interpolation
This version focuses on core functionality and will actually run
"""

import torch
import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm
from unet import FrameInterpolationUNet
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def preprocess_image(image_path, target_size=(256, 256)):
    """Preprocess image: resize to target size and normalize to [-1, 1]"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = 2.0 * image - 1.0
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image_tensor

def postprocess_image(tensor):
    """Postprocess tensor: convert from [-1, 1] to [0, 255] uint8"""
    image = (tensor + 1.0) / 2.0
    image = torch.clamp(image, 0.0, 1.0)
    image_np = image.squeeze().cpu().numpy()
    image_uint8 = (image_np * 255).astype(np.uint8)
    return image_uint8

def load_model(model_path, device):
    """Load trained U-Net model"""
    model = FrameInterpolationUNet(bilinear=True)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_path}")
        print(f"Trained for {checkpoint.get('epoch', 'Unknown')} epochs")
        print(f"Best validation loss: {checkpoint.get('val_loss', 'Unknown'):.6f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Model state dict loaded from {model_path}")
    
    model = model.to(device)
    model.eval()
    return model

def interpolate_frames(model, frame1, frame2, device):
    """Generate interpolated frame between two input frames using U-Net"""
    frame1 = frame1.to(device)
    frame2 = frame2.to(device)
    
    with torch.no_grad():
        interpolated = model(frame1, frame2)
    
    return interpolated

def linear_interpolation_baseline(frame1, frame2):
    """Linear interpolation baseline using pixel averaging"""
    interpolated = (frame1 + frame2) / 2.0
    return interpolated

def optical_flow_interpolation_baseline(frame1_np, frame2_np):
    """Optical flow interpolation baseline using OpenCV"""
    frame1_uint8 = frame1_np.astype(np.uint8)
    frame2_uint8 = frame2_np.astype(np.uint8)
    
    flow = cv2.calcOpticalFlowFarneback(
        frame1_uint8, frame2_uint8, None, pyr_scale=0.5, levels=3, 
        winsize=15, iterations=3, poly_n=5, poly_sigma=1.1, flags=0
    )
    
    flow_half = flow * 0.5
    h, w = frame1_uint8.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    new_x = x_coords + flow_half[:, :, 0]
    new_y = y_coords + flow_half[:, :, 1]
    
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)
    
    interpolated = cv2.remap(
        frame1_uint8, new_x, new_y, cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return interpolated

def compute_psnr(pred, target):
    """Compute PSNR between predicted and target images"""
    return psnr(target, pred, data_range=255)

def compute_ssim(pred, target):
    """Compute SSIM between predicted and target images"""
    return ssim(target, pred, data_range=255)

def load_test_triplets(test_dir):
    """Load test triplets from directory structure"""
    triplets = []
    
    for video_dir in os.listdir(test_dir):
        video_path = os.path.join(test_dir, video_dir)
        if os.path.isdir(video_path):
            frame_files = sorted([f for f in os.listdir(video_path) 
                               if f.endswith(('.jpg', '.png', '.bmp'))])
            
            for i in range(len(frame_files) - 2):
                triplet = {
                    'video_dir': video_path,
                    'frame_t0': frame_files[i],
                    'frame_t1': frame_files[i + 2],
                    'ground_truth': frame_files[i + 1],
                    'video_name': video_dir,
                    'triplet_id': i
                }
                triplets.append(triplet)
    
    return triplets

def evaluate_model_simple(model, test_triplets, device, save_results=False, output_dir=None):
    """Simplified evaluation that will actually work"""
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results storage
    methods = ['unet', 'linear', 'optical_flow']
    results_by_method = {method: [] for method in methods}
    metrics_by_method = {method: {'psnr': [], 'ssim': []} for method in methods}
    
    print(f"Evaluating {len(methods)} methods on {len(test_triplets)} test triplets...")
    
    for triplet in tqdm(test_triplets, desc="Processing triplets"):
        try:
            # Load frames
            frame_t0_path = os.path.join(triplet['video_dir'], triplet['frame_t0'])
            frame_t1_path = os.path.join(triplet['video_dir'], triplet['frame_t1'])
            ground_truth_path = os.path.join(triplet['video_dir'], triplet['ground_truth'])
            
            # Preprocess frames for U-Net
            frame_t0 = preprocess_image(frame_t0_path)
            frame_t1 = preprocess_image(frame_t1_path)
            
            # Load frames as numpy arrays for baseline methods
            frame_t0_np = cv2.imread(frame_t0_path, cv2.IMREAD_GRAYSCALE)
            frame_t1_np = cv2.imread(frame_t1_path, cv2.IMREAD_GRAYSCALE)
            frame_t0_np = cv2.resize(frame_t0_np, (256, 256))
            frame_t1_np = cv2.resize(frame_t1_np, (256, 256))
            
            # Load ground truth
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.resize(ground_truth, (256, 256))
            
            # Generate interpolated frames using all methods
            interpolated_frames = {}
            
            # U-Net method
            interpolated_unet = interpolate_frames(model, frame_t0, frame_t1, device)
            interpolated_frames['unet'] = postprocess_image(interpolated_unet)
            
            # Linear interpolation baseline
            interpolated_linear = linear_interpolation_baseline(frame_t0, frame_t1)
            interpolated_frames['linear'] = postprocess_image(interpolated_linear)
            
            # Optical flow baseline
            interpolated_optical_flow = optical_flow_interpolation_baseline(frame_t0_np, frame_t1_np)
            interpolated_frames['optical_flow'] = interpolated_optical_flow
            
            # Compute metrics for all methods
            for method in methods:
                interpolated_frame = interpolated_frames[method]
                
                psnr_val = compute_psnr(interpolated_frame, ground_truth)
                ssim_val = compute_ssim(interpolated_frame, ground_truth)
                
                metrics_by_method[method]['psnr'].append(psnr_val)
                metrics_by_method[method]['ssim'].append(ssim_val)
                
                result = {
                    'video_name': triplet['video_name'],
                    'triplet_id': triplet['triplet_id'],
                    'frame_t0': triplet['frame_t0'],
                    'frame_t1': triplet['frame_t1'],
                    'ground_truth': triplet['ground_truth'],
                    'method': method,
                    'psnr': psnr_val,
                    'ssim': ssim_val
                }
                results_by_method[method].append(result)
            
            # Save generated frames if requested
            if save_results and output_dir:
                for method in methods:
                    output_filename = f"{triplet['video_name']}_{triplet['triplet_id']:03d}_{method}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, interpolated_frames[method])
                
                gt_filename = f"{triplet['video_name']}_{triplet['triplet_id']:03d}_ground_truth.png"
                gt_path = os.path.join(output_dir, gt_filename)
                cv2.imwrite(gt_path, ground_truth)
                
        except Exception as e:
            print(f"Error processing triplet {triplet['video_name']}_{triplet['triplet_id']}: {e}")
            continue
    
    # Calculate comprehensive results
    evaluation_results = {
        'total_triplets': len(test_triplets),
        'successful_evaluations': len(results_by_method['unet']),
        'methods': methods,
        'results_by_method': results_by_method,
        'metrics_by_method': {}
    }
    
    # Calculate statistics for each method
    for method in methods:
        psnr_values = metrics_by_method[method]['psnr']
        ssim_values = metrics_by_method[method]['ssim']
        
        evaluation_results['metrics_by_method'][method] = {
            'average_psnr': np.mean(psnr_values) if psnr_values else 0.0,
            'average_ssim': np.mean(ssim_values) if ssim_values else 0.0,
            'std_psnr': np.std(psnr_values) if psnr_values else 0.0,
            'std_ssim': np.std(ssim_values) if ssim_values else 0.0,
            'min_psnr': np.min(psnr_values) if psnr_values else 0.0,
            'max_psnr': np.max(psnr_values) if psnr_values else 0.0,
            'min_ssim': np.min(ssim_values) if ssim_values else 0.0,
            'max_ssim': np.max(ssim_values) if ssim_values else 0.0
        }
    
    return evaluation_results

def print_simple_summary(results):
    """Print a simple summary of results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)
    print(f"Total test triplets: {results['total_triplets']}")
    print(f"Successful evaluations: {results['successful_evaluations']}")
    print()
    
    # Print results for each method
    for method in results['methods']:
        method_name = method.replace('_', ' ').title()
        metrics = results['metrics_by_method'][method]
        
        print(f"{method_name.upper()} METHOD:")
        print(f"  PSNR: {metrics['average_psnr']:.4f} ± {metrics['std_psnr']:.4f} dB")
        print(f"  SSIM: {metrics['average_ssim']:.4f} ± {metrics['std_ssim']:.4f}")
        print()
    
    # Print comparison
    print("METHOD COMPARISON:")
    print("-" * 40)
    baseline_psnr = results['metrics_by_method']['linear']['average_psnr']
    baseline_ssim = results['metrics_by_method']['linear']['average_ssim']
    
    for method in results['methods']:
        method_name = method.replace('_', ' ').title()
        metrics = results['metrics_by_method'][method]
        
        if method == 'linear':
            print(f"{method_name:<20} | Baseline")
        else:
            psnr_improvement = metrics['average_psnr'] - baseline_psnr
            ssim_improvement = metrics['average_ssim'] - baseline_ssim
            print(f"{method_name:<20} | PSNR: {psnr_improvement:+.2f} dB, SSIM: {ssim_improvement:+.4f}")

def save_simple_results(results, output_path):
    """Save results to JSON file"""
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json_results = json.loads(json.dumps(results, default=convert_numpy_types))
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Simple Frame Interpolation Evaluation')
    parser.add_argument('--test-dir', required=True, help='Directory containing test triplets')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save-results', action='store_true', help='Save generated frames')
    parser.add_argument('--output-dir', default='results', help='Directory to save results')
    parser.add_argument('--json-output', help='Path to save evaluation results as JSON')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # Load test triplets
        print(f"Loading test triplets from: {args.test_dir}")
        test_triplets = load_test_triplets(args.test_dir)
        
        if not test_triplets:
            print("No test triplets found. Please check your test directory structure.")
            return
        
        print(f"Found {len(test_triplets)} test triplets")
        
        # Load model
        print("Loading trained model...")
        model = load_model(args.model, device)
        
        # Evaluate model
        results = evaluate_model_simple(
            model, test_triplets, device, 
            save_results=args.save_results, output_dir=args.output_dir
        )
        
        # Print results
        print_simple_summary(results)
        
        # Save results
        if args.json_output:
            save_simple_results(results, args.json_output)
        
        if args.save_results:
            default_json_path = os.path.join(args.output_dir, 'evaluation_results.json')
            save_simple_results(results, default_json_path)
        
        print("\n✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
