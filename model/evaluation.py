import torch
import torch.nn.functional as F
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
    Generate interpolated frame between two input frames using U-Net
    
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

def linear_interpolation_baseline(frame1, frame2):
    """
    Linear interpolation baseline using pixel averaging
    
    Args:
        frame1: First frame tensor [1, 1, H, W] with values in [-1, 1]
        frame2: Second frame tensor [1, 1, H, W] with values in [-1, 1]
    
    Returns:
        Interpolated frame tensor [1, 1, H, W]
    """
    # Simple linear interpolation: (frame1 + frame2) / 2
    interpolated = (frame1 + frame2) / 2.0
    return interpolated

def optical_flow_interpolation_baseline(frame1_np, frame2_np):
    """
    Optical flow interpolation baseline using OpenCV
    
    Args:
        frame1_np: First frame as numpy array (H, W) with values in [0, 255]
        frame2_np: Second frame as numpy array (H, W) with values in [0, 255]
    
    Returns:
        Interpolated frame as numpy array (H, W) with values in [0, 255]
    """
    # Ensure frames are uint8
    frame1_uint8 = frame1_np.astype(np.uint8)
    frame2_uint8 = frame2_np.astype(np.uint8)
    
    # Calculate optical flow from frame1 to frame2
    flow = cv2.calcOpticalFlowFarneback(
        frame1_uint8, frame2_uint8, 
        None,  # No previous flow
        pyr_scale=0.5,  # Pyramid scale
        levels=3,        # Number of pyramid levels
        winsize=15,      # Window size
        iterations=3,    # Number of iterations
        poly_n=5,        # Polynomial degree
        poly_sigma=1.1,  # Gaussian sigma
        flags=0          # Flags
    )
    
    # Calculate intermediate flow (halfway between frames)
    flow_half = flow * 0.5
    
    # Warp frame1 using the intermediate flow
    h, w = frame1_uint8.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Apply flow displacement
    new_x = x_coords + flow_half[:, :, 0]
    new_y = y_coords + flow_half[:, :, 1]
    
    # Ensure coordinates are within bounds
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)
    
    # Interpolate frame1 at new positions
    interpolated = cv2.remap(
        frame1_uint8, new_x, new_y, 
        cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return interpolated

def compute_psnr(pred, target):
    """
    Compute PSNR between predicted and target images
    
    Args:
        pred: Predicted image (numpy array, uint8)
        target: Target image (numpy array, uint8)
    
    Returns:
        PSNR value
    """
    return psnr(target, pred, data_range=255)

def compute_ssim(pred, target):
    """
    Compute SSIM between predicted and target images
    
    Args:
        pred: Predicted image (numpy array, uint8)
        target: Target image (numpy array, uint8)
    
    Returns:
        SSIM value
    """
    return ssim(target, pred, data_range=255)

def load_test_triplets(test_dir):
    """
    Load test triplets from directory structure
    
    Expected structure:
    test_dir/
    ├── video1/
    │   ├── frame_000.jpg
    │   ├── frame_001.jpg
    │   ├── frame_002.jpg
    │   └── ...
    ├── video2/
    │   └── ...
    
    Args:
        test_dir: Directory containing test data
    
    Returns:
        List of triplets with paths
    """
    triplets = []
    
    for video_dir in os.listdir(test_dir):
        video_path = os.path.join(test_dir, video_dir)
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
                    'ground_truth': frame_files[i + 1],  # Middle frame (target)
                    'video_name': video_dir,
                    'triplet_id': i
                }
                triplets.append(triplet)
    
    return triplets

def evaluate_model(model, test_triplets, device, save_results=False, output_dir=None):
    """
    Evaluate model on test triplets
    
    Args:
        model: Trained U-Net model
        test_triplets: List of test triplets
        device: Device to run inference on
        save_results: Whether to save generated frames
        output_dir: Directory to save results
    
    Returns:
        Dictionary with evaluation metrics
    """
    if save_results and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    psnr_values = []
    ssim_values = []
    results = []
    
    print(f"Evaluating model on {len(test_triplets)} test triplets...")
    
    for triplet in tqdm(test_triplets, desc="Processing triplets"):
        try:
            # Load frames
            frame_t0_path = os.path.join(triplet['video_dir'], triplet['frame_t0'])
            frame_t1_path = os.path.join(triplet['video_dir'], triplet['frame_t1'])
            ground_truth_path = os.path.join(triplet['video_dir'], triplet['ground_truth'])
            
            # Preprocess frames
            frame_t0 = preprocess_image(frame_t0_path)
            frame_t1 = preprocess_image(frame_t1_path)
            
            # Generate interpolated frame
            interpolated = interpolate_frames(model, frame_t0, frame_t1, device)
            
            # Postprocess interpolated frame
            interpolated_uint8 = postprocess_image(interpolated)
            
            # Load ground truth for comparison
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.resize(ground_truth, (256, 256))
            
            # Compute metrics
            psnr_val = compute_psnr(interpolated_uint8, ground_truth)
            ssim_val = compute_ssim(interpolated_uint8, ground_truth)
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            
            # Store results
            result = {
                'video_name': triplet['video_name'],
                'triplet_id': triplet['triplet_id'],
                'frame_t0': triplet['frame_t0'],
                'frame_t1': triplet['frame_t1'],
                'ground_truth': triplet['ground_truth'],
                'psnr': psnr_val,
                'ssim': ssim_val
            }
            results.append(result)
            
            # Save generated frame if requested
            if save_results and output_dir:
                output_filename = f"{triplet['video_name']}_{triplet['triplet_id']:03d}_generated.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, interpolated_uint8)
                
                # Also save ground truth for comparison
                gt_filename = f"{triplet['video_name']}_{triplet['triplet_id']:03d}_ground_truth.png"
                gt_path = os.path.join(output_dir, gt_filename)
                cv2.imwrite(gt_path, ground_truth)
                
        except Exception as e:
            print(f"Error processing triplet {triplet['video_name']}_{triplet['triplet_id']}: {e}")
            continue
    
    # Compute average metrics
    avg_psnr = np.mean(psnr_values) if psnr_values else 0.0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0.0
    std_psnr = np.std(psnr_values) if psnr_values else 0.0
    std_ssim = np.std(ssim_values) if ssim_values else 0.0
    
    evaluation_results = {
        'total_triplets': len(test_triplets),
        'successful_evaluations': len(results),
        'average_psnr': avg_psnr,
        'average_ssim': avg_ssim,
        'std_psnr': std_psnr,
        'std_ssim': std_ssim,
        'min_psnr': np.min(psnr_values) if psnr_values else 0.0,
        'max_psnr': np.max(psnr_values) if psnr_values else 0.0,
        'min_ssim': np.min(ssim_values) if ssim_values else 0.0,
        'max_ssim': np.max(ssim_values) if ssim_values else 0.0,
        'detailed_results': results
    }
    
    return evaluation_results

def print_evaluation_summary(results):
    """
    Print evaluation results summary for all methods
    
    Args:
        results: Dictionary with evaluation results for all methods
    """
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY - ALL METHODS")
    print("="*80)
    print(f"Total test triplets: {results['total_triplets']}")
    print(f"Successful evaluations: {results['successful_evaluations']}")
    print()
    
    # Print results for each method
    for method in results['methods']:
        method_name = method.replace('_', ' ').title()
        metrics = results['metrics_by_method'][method]
        
        print(f"{method_name.upper()} METHOD:")
        print("-" * 50)
        print(f"PSNR (Peak Signal-to-Noise Ratio):")
        print(f"  Average: {metrics['average_psnr']:.4f} dB")
        print(f"  Std Dev: {metrics['std_psnr']:.4f} dB")
        print(f"  Range:   {metrics['min_psnr']:.4f} - {metrics['max_psnr']:.4f} dB")
        print()
        print(f"SSIM (Structural Similarity Index):")
        print(f"  Average: {metrics['average_ssim']:.4f}")
        print(f"  Std Dev: {metrics['std_ssim']:.4f}")
        print(f"  Range:   {metrics['min_ssim']:.4f} - {metrics['max_ssim']:.4f}")
        print()
    
    # Print method comparison
    print("METHOD COMPARISON:")
    print("=" * 50)
    print(f"{'Method':<20} | {'PSNR (dB)':<12} | {'SSIM':<8} | {'Improvement':<15}")
    print("-" * 70)
    
    # Get baseline method (linear interpolation)
    baseline_psnr = results['metrics_by_method']['linear']['average_psnr']
    baseline_ssim = results['metrics_by_method']['linear']['average_ssim']
    
    for method in results['methods']:
        method_name = method.replace('_', ' ').title()
        metrics = results['metrics_by_method'][method]
        
        # Calculate improvement over baseline
        if method == 'linear':
            psnr_improvement = "Baseline"
            ssim_improvement = "Baseline"
        else:
            psnr_improvement = f"+{metrics['average_psnr'] - baseline_psnr:+.2f} dB"
            ssim_improvement = f"+{metrics['average_ssim'] - baseline_ssim:+.4f}"
        
        print(f"{method_name:<20} | {metrics['average_psnr']:<12.4f} | {metrics['average_ssim']:<8.4f} | {psnr_improvement:<15}")
    
    print()
    
    # Print per-video breakdown for U-Net method
    if results['results_by_method']['unet']:
        print("PER-VIDEO BREAKDOWN (U-Net Method):")
        print("-" * 50)
        
        video_metrics = {}
        for result in results['results_by_method']['unet']:
            video_name = result['video_name']
            if video_name not in video_metrics:
                video_metrics[video_name] = {'psnr': [], 'ssim': []}
            video_metrics[video_name]['psnr'].append(result['psnr'])
            video_metrics[video_name]['ssim'].append(result['ssim'])
        
        for video_name, metrics in video_metrics.items():
            avg_video_psnr = np.mean(metrics['psnr'])
            avg_video_ssim = np.mean(metrics['ssim'])
            print(f"{video_name:20s} | PSNR: {avg_video_psnr:6.4f} | SSIM: {avg_video_ssim:6.4f}")
    
    print()
    
    # Print statistical significance analysis
    print("STATISTICAL ANALYSIS:")
    print("-" * 30)
    
    # Compare U-Net vs Linear
    unet_psnr = results['metrics_by_method']['unet']['average_psnr']
    linear_psnr = results['metrics_by_method']['linear']['average_psnr']
    unet_ssim = results['metrics_by_method']['unet']['average_ssim']
    linear_ssim = results['metrics_by_method']['linear']['average_ssim']
    
    print(f"U-Net vs Linear Interpolation:")
    print(f"  PSNR improvement: {unet_psnr - linear_psnr:+.4f} dB")
    print(f"  SSIM improvement: {unet_ssim - linear_ssim:+.4f}")
    
    # Compare U-Net vs Optical Flow
    optical_flow_psnr = results['metrics_by_method']['optical_flow']['average_psnr']
    optical_flow_ssim = results['metrics_by_method']['optical_flow']['average_ssim']
    
    print(f"U-Net vs Optical Flow:")
    print(f"  PSNR improvement: {unet_psnr - optical_flow_psnr:+.4f} dB")
    print(f"  SSIM improvement: {unet_ssim - optical_flow_ssim:+.4f}")

def save_evaluation_results(results, output_path):
    """
    Save evaluation results to JSON file
    
    Args:
        results: Dictionary with evaluation results for all methods
        output_path: Path to save JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Deep copy and convert
    json_results = json.loads(json.dumps(results, default=convert_numpy_types))
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Evaluation results saved to: {output_path}")
    
    # Also save a summary CSV file
    csv_path = output_path.replace('.json', '_summary.csv')
    try:
        import pandas as pd
        
        # Create summary dataframe
        summary_data = []
        for method in results['methods']:
            method_name = method.replace('_', ' ').title()
            metrics = results['metrics_by_method'][method]
            
            summary_data.append({
                'Method': method_name,
                'Average_PSNR_dB': metrics['average_psnr'],
                'Std_PSNR_dB': metrics['std_psnr'],
                'Min_PSNR_dB': metrics['min_psnr'],
                'Max_PSNR_dB': metrics['max_psnr'],
                'Average_SSIM': metrics['average_ssim'],
                'Std_SSIM': metrics['std_ssim'],
                'Min_SSIM': metrics['min_ssim'],
                'Max_SSIM': metrics['max_ssim']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")
        
    except ImportError:
        print("pandas not available, skipping CSV export")

def create_comparison_plots(results, output_dir):
    """
    Create comprehensive comparison plots for all methods
    
    Args:
        results: Dictionary with evaluation results for all methods
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style and backend
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Also ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Extract data for plotting
        methods = results['methods']
        method_names = [method.replace('_', ' ').title() for method in methods]
        
        psnr_data = [results['metrics_by_method'][method]['average_psnr'] for method in methods]
        ssim_data = [results['metrics_by_method'][method]['average_ssim'] for method in methods]
        
        # Set figure parameters
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        
        # 1. PSNR Comparison Bar Chart
        plt.figure(figsize=(10, 6))
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = plt.bar(method_names, psnr_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        plt.title('PSNR Comparison Across Methods', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        plt.xlabel('Method', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, value in zip(bars, psnr_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SSIM Comparison Bar Chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(method_names, ssim_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        plt.title('SSIM Comparison Across Methods', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('SSIM', fontsize=12, fontweight='bold')
        plt.xlabel('Method', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, value in zip(bars, ssim_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ssim_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Combined Metrics Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PSNR subplot
        bars1 = ax1.bar(method_names, psnr_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('PSNR Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # SSIM subplot
        bars2 = ax2.bar(method_names, ssim_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('SSIM Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.suptitle('Frame Interpolation Methods Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'methods_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Error Bar Plot with Standard Deviations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # PSNR with error bars
        psnr_stds = [results['metrics_by_method'][method]['std_psnr'] for method in methods]
        ax1.errorbar(method_names, psnr_data, yerr=psnr_stds, fmt='o', capsize=5, capthick=2, 
                    markersize=8, linewidth=2, color=colors)
        ax1.set_title('PSNR Comparison with Standard Deviation', fontsize=14, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # SSIM with error bars
        ssim_stds = [results['metrics_by_method'][method]['std_ssim'] for method in methods]
        ax2.errorbar(method_names, ssim_data, yerr=ssim_stds, fmt='o', capsize=5, capthick=2, 
                    markersize=8, linewidth=2, color=colors)
        ax2.set_title('SSIM Comparison with Standard Deviation', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Metrics Comparison with Error Bars', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_with_error_bars.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to: {output_dir}")
        
        # Create a summary README file
        create_plots_summary(output_dir)
        
    except ImportError:
        print("matplotlib/seaborn not available, skipping plot generation")
    except Exception as e:
        print(f"Error creating plots: {e}")

def create_plots_summary(output_dir):
    """
    Create a summary README file describing all generated plots
    
    Args:
        output_dir: Directory containing the plots
    """
    try:
        output_dir = Path(output_dir)
        readme_path = output_dir / "README_plots.md"
        
        with open(readme_path, 'w') as f:
            f.write("# Frame Interpolation Evaluation Plots\n\n")
            f.write("This directory contains comprehensive visualization plots for the frame interpolation evaluation.\n\n")
            
            f.write("## 📊 Metrics Comparison Plots\n\n")
            f.write("### 1. PSNR Comparison (`psnr_comparison.png`)\n")
            f.write("- Bar chart comparing Peak Signal-to-Noise Ratio across all methods\n")
            f.write("- Higher values indicate better image quality\n")
            f.write("- Methods: U-Net, Linear Interpolation, Optical Flow\n\n")
            
            f.write("### 2. SSIM Comparison (`ssim_comparison.png`)\n")
            f.write("- Bar chart comparing Structural Similarity Index across all methods\n")
            f.write("- Values range from 0 to 1, higher is better\n")
            f.write("- Measures perceptual similarity to ground truth\n\n")
            
            f.write("### 3. Combined Metrics (`methods_comparison.png`)\n")
            f.write("- Side-by-side comparison of PSNR and SSIM\n")
            f.write("- Easy visual comparison of both metrics\n\n")
            
            f.write("### 4. Error Bars (`metrics_with_error_bars.png`)\n")
            f.write("- PSNR and SSIM with standard deviation error bars\n")
            f.write("- Shows statistical variability across test samples\n\n")
            
            f.write("## 🖼️ Frame Comparison Plots\n\n")
            f.write("### Frame Comparison Examples (`frame_comparison_example_*.png`)\n")
            f.write("- Side-by-side comparison of input frames, ground truth, and generated frames\n")
            f.write("- Shows visual quality differences between methods\n")
            f.write("- Includes PSNR and SSIM metrics for each method\n\n")
            
            f.write("## 📁 File Structure\n\n")
            f.write("```\n")
            f.write(f"{output_dir.name}/\n")
            f.write("├── README_plots.md           # This file\n")
            f.write("├── psnr_comparison.png       # PSNR bar chart\n")
            f.write("├── ssim_comparison.png       # SSIM bar chart\n")
            f.write("├── methods_comparison.png    # Combined metrics\n")
            f.write("├── metrics_with_error_bars.png # Error bar plots\n")
            f.write("├── frame_comparison_example_1_*.png # Best example\n")
            f.write("├── frame_comparison_example_2_*.png # Middle example\n")
            f.write("└── frame_comparison_example_3_*.png # Worst example\n")
            f.write("```\n\n")
            
            f.write("## 🔍 How to Interpret\n\n")
            f.write("### PSNR (Peak Signal-to-Noise Ratio)\n")
            f.write("- **Excellent**: > 30 dB\n")
            f.write("- **Good**: 25-30 dB\n")
            f.write("- **Acceptable**: 20-25 dB\n")
            f.write("- **Poor**: < 20 dB\n\n")
            
            f.write("### SSIM (Structural Similarity Index)\n")
            f.write("- **Excellent**: > 0.95\n")
            f.write("- **Good**: 0.90-0.95\n")
            f.write("- **Acceptable**: 0.80-0.90\n")
            f.write("- **Poor**: < 0.80\n\n")
            
            f.write("## 📈 Usage Tips\n\n")
            f.write("1. **Compare Methods**: Use the bar charts to see which method performs best\n")
            f.write("2. **Assess Quality**: Check if PSNR > 25 dB and SSIM > 0.90 for good results\n")
            f.write("3. **Visual Inspection**: Examine frame comparison plots for perceptual quality\n")
            f.write("4. **Statistical Significance**: Use error bars to assess method reliability\n")
            f.write("5. **Export**: All plots are high-resolution (300 DPI) suitable for publications\n\n")
            
            f.write("---\n")
            f.write("*Generated automatically by the Frame Interpolation Evaluation Script*\n")
        
        print(f"Plots summary README created: {readme_path}")
        
    except Exception as e:
        print(f"Error creating plots summary: {e}")

def create_evaluation_report(results, output_dir):
    """
    Create a comprehensive evaluation report
    
    Args:
        results: Dictionary with evaluation results for all methods
        output_dir: Directory to save the report
    """
    try:
        output_dir = Path(output_dir)
        report_path = output_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Frame Interpolation Evaluation Report\n\n")
            f.write(f"**Generated**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 📋 Executive Summary\n\n")
            f.write(f"- **Total Test Triplets**: {results['total_triplets']}\n")
            f.write(f"- **Successful Evaluations**: {results['successful_evaluations']}\n")
            f.write(f"- **Evaluation Methods**: {', '.join(results['methods']).replace('_', ' ').title()}\n\n")
            
            f.write("## 🏆 Method Rankings\n\n")
            
            # Rank methods by PSNR
            methods_psnr = [(method, results['metrics_by_method'][method]['average_psnr']) 
                           for method in results['methods']]
            methods_psnr.sort(key=lambda x: x[1], reverse=True)
            
            f.write("### PSNR Ranking (Best to Worst)\n")
            for i, (method, psnr) in enumerate(methods_psnr, 1):
                method_name = method.replace('_', ' ').title()
                f.write(f"{i}. **{method_name}**: {psnr:.4f} dB\n")
            f.write("\n")
            
            # Rank methods by SSIM
            methods_ssim = [(method, results['metrics_by_method'][method]['average_ssim']) 
                           for method in results['methods']]
            methods_ssim.sort(key=lambda x: x[1], reverse=True)
            
            f.write("### SSIM Ranking (Best to Worst)\n")
            for i, (method, ssim) in enumerate(methods_ssim, 1):
                method_name = method.replace('_', ' ').title()
                f.write(f"{i}. **{method_name}**: {ssim:.4f}\n")
            f.write("\n")
            
            f.write("## 📊 Detailed Metrics\n\n")
            
            for method in results['methods']:
                method_name = method.replace('_', ' ').title()
                metrics = results['metrics_by_method'][method]
                
                f.write(f"### {method_name} Method\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                f.write(f"| Average PSNR | {metrics['average_psnr']:.4f} dB |\n")
                f.write(f"| PSNR Std Dev | {metrics['std_psnr']:.4f} dB |\n")
                f.write(f"| PSNR Range | {metrics['min_psnr']:.4f} - {metrics['max_psnr']:.4f} dB |\n")
                f.write(f"| Average SSIM | {metrics['average_ssim']:.4f} |\n")
                f.write(f"| SSIM Std Dev | {metrics['std_ssim']:.4f} |\n")
                f.write(f"| SSIM Range | {metrics['min_ssim']:.4f} - {metrics['max_ssim']:.4f} |\n\n")
            
            f.write("## 🔍 Performance Analysis\n\n")
            
            # Get baseline (linear interpolation)
            baseline_psnr = results['metrics_by_method']['linear']['average_psnr']
            baseline_ssim = results['metrics_by_method']['linear']['average_ssim']
            
            f.write("### Improvement Over Linear Interpolation (Baseline)\n\n")
            f.write("| Method | PSNR Improvement | SSIM Improvement |\n")
            f.write("|--------|------------------|------------------|\n")
            
            for method in results['methods']:
                if method == 'linear':
                    f.write(f"| {method.replace('_', ' ').title()} | Baseline | Baseline |\n")
                else:
                    method_psnr = results['metrics_by_method'][method]['average_psnr']
                    method_ssim = results['metrics_by_method'][method]['average_ssim']
                    psnr_improvement = method_psnr - baseline_psnr
                    ssim_improvement = method_ssim - baseline_ssim
                    f.write(f"| {method.replace('_', ' ').title()} | {psnr_improvement:+.4f} dB | {ssim_improvement:+.4f} |\n")
            
            f.write("\n")
            
            f.write("## 📈 Quality Assessment\n\n")
            
            # Assess overall quality
            best_psnr = max(results['metrics_by_method'][method]['average_psnr'] for method in results['methods'])
            best_ssim = max(results['metrics_by_method'][method]['average_ssim'] for method in results['methods'])
            
            f.write("### Overall Performance Assessment\n\n")
            
            if best_psnr > 30:
                f.write("- **PSNR Quality**: 🟢 Excellent (> 30 dB)\n")
            elif best_psnr > 25:
                f.write("- **PSNR Quality**: 🟡 Good (25-30 dB)\n")
            elif best_psnr > 20:
                f.write("- **PSNR Quality**: 🟠 Acceptable (20-25 dB)\n")
            else:
                f.write("- **PSNR Quality**: 🔴 Poor (< 20 dB)\n")
            
            if best_ssim > 0.95:
                f.write("- **SSIM Quality**: 🟢 Excellent (> 0.95)\n")
            elif best_ssim > 0.90:
                f.write("- **SSIM Quality**: 🟡 Good (0.90-0.95)\n")
            elif best_ssim > 0.80:
                f.write("- **SSIM Quality**: 🟠 Acceptable (0.80-0.90)\n")
            else:
                f.write("- **SSIM Quality**: 🔴 Poor (< 0.80)\n")
            
            f.write("\n")
            
            f.write("## 🎯 Recommendations\n\n")
            
            # Generate recommendations based on results
            best_method_psnr = methods_psnr[0][0]
            best_method_ssim = methods_ssim[0][0]
            
            f.write("### Based on Current Results:\n\n")
            
            if best_method_psnr == best_method_ssim:
                f.write(f"- **Primary Recommendation**: Use **{best_method_psnr.replace('_', ' ').title()}** method\n")
                f.write("  - Best performance on both PSNR and SSIM metrics\n")
            else:
                f.write(f"- **For PSNR Optimization**: Use **{best_method_psnr.replace('_', ' ').title()}** method\n")
                f.write(f"- **For SSIM Optimization**: Use **{best_method_ssim.replace('_', ' ').title()}** method\n")
            
            f.write("\n")
            
            # Check if U-Net is performing well
            if 'unet' in results['methods']:
                unet_psnr = results['metrics_by_method']['unet']['average_psnr']
                unet_ssim = results['metrics_by_method']['unet']['average_ssim']
                
                if unet_psnr > baseline_psnr + 2 and unet_ssim > baseline_ssim + 0.05:
                    f.write("- **U-Net Performance**: 🟢 Excellent - significantly outperforms baselines\n")
                elif unet_psnr > baseline_psnr + 1 and unet_ssim > baseline_ssim + 0.02:
                    f.write("- **U-Net Performance**: 🟡 Good - moderately outperforms baselines\n")
                elif unet_psnr > baseline_psnr and unet_ssim > baseline_ssim:
                    f.write("- **U-Net Performance**: 🟠 Acceptable - slightly outperforms baselines\n")
                else:
                    f.write("- **U-Net Performance**: 🔴 Poor - consider retraining or architecture changes\n")
                
                f.write("\n")
            
            f.write("## 📁 Generated Files\n\n")
            f.write("The following files were generated during evaluation:\n\n")
            f.write("- **Plots**: PSNR/SSIM comparisons, frame examples\n")
            f.write("- **Data**: JSON results, CSV summaries\n")
            f.write("- **Documentation**: This report and plot descriptions\n\n")
            
            f.write("## 🔧 Next Steps\n\n")
            f.write("1. **Review Plots**: Examine visual comparisons in the generated plots\n")
            f.write("2. **Analyze Results**: Use metrics to identify areas for improvement\n")
            f.write("3. **Model Optimization**: Consider hyperparameter tuning if U-Net underperforms\n")
            f.write("4. **Data Quality**: Assess if training data quality affects results\n")
            f.write("5. **Architecture**: Explore alternative network architectures if needed\n\n")
            
            f.write("---\n")
            f.write("*This report was generated automatically by the Frame Interpolation Evaluation Script*\n")
        
        print(f"Comprehensive evaluation report created: {report_path}")
        
    except Exception as e:
        print(f"Error creating evaluation report: {e}")

def create_frame_comparison_plots(results, output_dir, num_examples=3):
    """
    Create side-by-side frame comparison plots
    
    Args:
        results: Dictionary with evaluation results for all methods
        output_dir: Directory to save plots
        num_examples: Number of example triplets to visualize
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Also ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Set figure parameters
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 8
        
        # Get example triplets (best, worst, and random)
        if 'results_by_method' not in results or 'unet' not in results['results_by_method']:
            print("No detailed results available for frame comparison plots")
            return
        
        unet_results = results['results_by_method']['unet']
        if not unet_results:
            print("No U-Net results available for frame comparison plots")
            return
        
        # Sort by PSNR to get best and worst examples
        unet_results_sorted = sorted(unet_results, key=lambda x: x['psnr'], reverse=True)
        
        # Select examples: best, worst, and middle
        examples = []
        if len(unet_results_sorted) >= 3:
            examples = [
                unet_results_sorted[0],  # Best
                unet_results_sorted[len(unet_results_sorted)//2],  # Middle
                unet_results_sorted[-1]  # Worst
            ]
        elif len(unet_results_sorted) >= 1:
            examples = unet_results_sorted[:min(num_examples, len(unet_results_sorted))]
        
        if not examples:
            print("No examples available for frame comparison plots")
            return
        
        # Create frame comparison plots
        for i, example in enumerate(examples):
            try:
                # Load the frames
                frame_t0_path = os.path.join(example['video_dir'], example['frame_t0'])
                frame_t1_path = os.path.join(example['video_dir'], example['frame_t1'])
                ground_truth_path = os.path.join(example['video_dir'], example['ground_truth'])
                
                # Read frames
                frame_t0 = cv2.imread(frame_t0_path, cv2.IMREAD_GRAYSCALE)
                frame_t1 = cv2.imread(frame_t1_path, cv2.IMREAD_GRAYSCALE)
                ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize to 256x256 if needed
                if frame_t0.shape != (256, 256):
                    frame_t0 = cv2.resize(frame_t0, (256, 256))
                if frame_t1.shape != (256, 256):
                    frame_t1 = cv2.resize(frame_t1, (256, 256))
                if ground_truth.shape != (256, 256):
                    ground_truth = cv2.resize(ground_truth, (256, 256))
                
                # Create the comparison plot
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                fig.suptitle(f'Frame Interpolation Comparison - Example {i+1}\n'
                           f'Video: {example["video_name"]}, Triplet: {example["triplet_id"]}\n'
                           f'U-Net PSNR: {example["psnr"]:.2f} dB, SSIM: {example["ssim"]:.4f}', 
                           fontsize=14, fontweight='bold', y=0.98)
                
                # Row 1: Input frames and ground truth
                axes[0, 0].imshow(frame_t0, cmap='gray')
                axes[0, 0].set_title('Frame T0 (Input)', fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(frame_t1, cmap='gray')
                axes[0, 1].set_title('Frame T1 (Input)', fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(ground_truth, cmap='gray')
                axes[0, 2].set_title('Ground Truth (Target)', fontweight='bold')
                axes[0, 2].axis('off')
                
                # Empty subplot for spacing
                axes[0, 3].axis('off')
                
                # Row 2: Generated frames from different methods
                method_colors = {'unet': '#2E86AB', 'linear': '#A23B72', 'optical_flow': '#F18F01'}
                
                for j, method in enumerate(['unet', 'linear', 'optical_flow']):
                    if method in results['results_by_method']:
                        # Find the corresponding result for this triplet
                        method_result = None
                        for result in results['results_by_method'][method]:
                            if (result['video_name'] == example['video_name'] and 
                                result['triplet_id'] == example['triplet_id']):
                                method_result = result
                                break
                        
                        if method_result:
                            # Load the generated frame
                            if 'save_results' in results and results.get('save_results'):
                                # Try to load saved frame
                                generated_path = os.path.join(output_dir, 
                                    f"{example['video_name']}_{example['triplet_id']:03d}_{method}.png")
                                if os.path.exists(generated_path):
                                    generated_frame = cv2.imread(generated_path, cv2.IMREAD_GRAYSCALE)
                                else:
                                    # Generate frame on-the-fly for visualization
                                    generated_frame = generate_frame_for_plot(method, frame_t0, frame_t1, results)
                            else:
                                # Generate frame on-the-fly
                                generated_frame = generate_frame_for_plot(method, frame_t0, frame_t1, results)
                            
                            if generated_frame is not None:
                                axes[1, j].imshow(generated_frame, cmap='gray')
                                method_name = method.replace('_', ' ').title()
                                axes[1, j].set_title(f'{method_name}\nPSNR: {method_result["psnr"]:.2f} dB\nSSIM: {method_result["ssim"]:.4f}', 
                                                   fontweight='bold', fontsize=9)
                                axes[1, j].axis('off')
                            else:
                                axes[1, j].text(0.5, 0.5, f'{method_name}\nFrame not available', 
                                              ha='center', va='center', transform=axes[1, j].transAxes,
                                              fontweight='bold')
                                axes[1, j].axis('off')
                        else:
                            axes[1, j].text(0.5, 0.5, f'{method.replace("_", " ").title()}\nNo result', 
                                          ha='center', va='center', transform=axes[1, j].transAxes,
                                          fontweight='bold')
                            axes[1, j].axis('off')
                
                # Empty subplot for spacing
                axes[1, 3].axis('off')
                
                plt.tight_layout()
                plt.subplots_adjust(top=0.9)
                
                # Save the plot
                plot_filename = f'frame_comparison_example_{i+1}_{example["video_name"]}_{example["triplet_id"]:03d}.png'
                plt.savefig(os.path.join(output_dir, plot_filename), dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Frame comparison plot {i+1} saved: {plot_filename}")
                
            except Exception as e:
                print(f"Error creating frame comparison plot {i+1}: {e}")
                continue
        
        print(f"Frame comparison plots saved to: {output_dir}")
        
    except ImportError:
        print("matplotlib not available, skipping frame comparison plots")
    except Exception as e:
        print(f"Error creating frame comparison plots: {e}")

def generate_frame_for_plot(method, frame_t0, frame_t1, results):
    """
    Generate a frame for plotting purposes (fallback when saved frames aren't available)
    
    Args:
        method: Method name ('unet', 'linear', 'optical_flow')
        frame_t0: First frame as numpy array
        frame_t1: Second frame as numpy array
        results: Evaluation results dictionary
    
    Returns:
        Generated frame as numpy array or None
    """
    try:
        if method == 'linear':
            # Simple linear interpolation
            return ((frame_t0.astype(np.float32) + frame_t1.astype(np.float32)) / 2).astype(np.uint8)
        elif method == 'optical_flow':
            # Optical flow interpolation
            return optical_flow_interpolation_baseline(frame_t0, frame_t1)
        else:
            # For U-Net, we can't generate without the model, so return None
            return None
    except Exception as e:
        print(f"Error generating frame for {method}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Evaluate Frame Interpolation Model')
    parser.add_argument('--test-dir', required=True, help='Directory containing test triplets')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save-results', action='store_true', help='Save generated frames')
    parser.add_argument('--output-dir', default='results', help='Directory to save results')
    parser.add_argument('--json-output', help='Path to save evaluation results as JSON')
    parser.add_argument('--generate-plots', action='store_true', default=True, help='Generate comparison plots (default: True)')
    parser.add_argument('--frame-comparisons', action='store_true', default=True, help='Generate frame comparison plots (default: True)')
    parser.add_argument('--num-examples', type=int, default=3, help='Number of frame comparison examples')
    
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
        results = evaluate_model(
            model, 
            test_triplets, 
            device, 
            save_results=args.save_results,
            output_dir=args.output_dir
        )
        
        # Print results
        print_evaluation_summary(results)
        
        # Save results to JSON if requested
        if args.json_output:
            save_evaluation_results(results, args.json_output)
        
        # Save results to default location if saving frames
        if args.save_results:
            default_json_path = os.path.join(args.output_dir, 'evaluation_results.json')
            save_evaluation_results(results, default_json_path)
        
        # Generate comparison plots if requested (default to True for comprehensive evaluation)
        if args.generate_plots:
            create_comparison_plots(results, args.output_dir)
        
        # Generate frame comparison plots if requested (default to True for comprehensive evaluation)
        if args.frame_comparisons:
            create_frame_comparison_plots(results, args.output_dir, args.num_examples)
        
        # Create comprehensive evaluation report
        create_evaluation_report(results, args.output_dir)
        
        print("\nEvaluation completed successfully!")
        print(f"\n📊 All plots and results saved to: {args.output_dir}/")
        print("📁 Check the README_plots.md file for detailed plot descriptions!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
