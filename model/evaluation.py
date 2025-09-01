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
    Create comparison plots for all methods
    
    Args:
        results: Dictionary with evaluation results for all methods
        output_dir: Directory to save plots
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        methods = results['methods']
        method_names = [method.replace('_', ' ').title() for method in methods]
        
        psnr_data = [results['metrics_by_method'][method]['average_psnr'] for method in methods]
        ssim_data = [results['metrics_by_method'][method]['average_ssim'] for method in methods]
        
        # Create PSNR comparison plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(method_names, psnr_data, color=['#2E86AB', '#A23B72', '#F18F01'])
        plt.title('PSNR Comparison Across Methods', fontsize=16, fontweight='bold')
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, psnr_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create SSIM comparison plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(method_names, ssim_data, color=['#2E86AB', '#A23B72', '#F18F01'])
        plt.title('SSIM Comparison Across Methods', fontsize=16, fontweight='bold')
        plt.ylabel('SSIM', fontsize=12)
        plt.xlabel('Method', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, ssim_data):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ssim_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create combined metrics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PSNR subplot
        bars1 = ax1.bar(method_names, psnr_data, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax1.set_title('PSNR Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('PSNR (dB)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # SSIM subplot
        bars2 = ax2.bar(method_names, ssim_data, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax2.set_title('SSIM Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Frame Interpolation Methods Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'methods_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plots saved to: {output_dir}")
        
    except ImportError:
        print("matplotlib/seaborn not available, skipping plot generation")
    except Exception as e:
        print(f"Error creating plots: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Frame Interpolation Model')
    parser.add_argument('--test-dir', required=True, help='Directory containing test triplets')
    parser.add_argument('--model', default='best_model.pth', help='Path to trained model')
    parser.add_argument('--device', default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save-results', action='store_true', help='Save generated frames')
    parser.add_argument('--output-dir', default='evaluation_results', help='Directory to save results')
    parser.add_argument('--json-output', help='Path to save evaluation results as JSON')
    parser.add_argument('--generate-plots', action='store_true', help='Generate comparison plots')
    
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
        
        # Generate comparison plots if requested
        if args.generate_plots:
            create_comparison_plots(results, args.output_dir)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
