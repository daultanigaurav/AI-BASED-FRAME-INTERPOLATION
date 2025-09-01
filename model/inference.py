import torch
import cv2
import numpy as np
import os
from unet import FrameInterpolationUNet
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class FrameInterpolator:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = FrameInterpolationUNet(bilinear=True)
        
        # Load trained model
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame for model input"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size (256x256)
        frame_resized = cv2.resize(frame_rgb, (256, 256))
        
        # Normalize to [0, 1]
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return frame_tensor
    
    def postprocess_frame(self, frame_tensor):
        """Convert model output back to image format"""
        # Remove batch dimension and convert to numpy
        frame_np = frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Clip values to [0, 1]
        frame_np = np.clip(frame_np, 0, 1)
        
        # Convert to uint8 [0, 255]
        frame_uint8 = (frame_np * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def interpolate_frames(self, frame1, frame2):
        """Interpolate between two frames to generate an intermediate frame"""
        # Preprocess frames
        frame1_tensor = self.preprocess_frame(frame1).to(self.device)
        frame2_tensor = self.preprocess_frame(frame2).to(self.device)
        
        with torch.no_grad():
            # Generate intermediate frame
            intermediate_tensor = self.model(frame1_tensor, frame2_tensor)
        
        # Postprocess output
        intermediate_frame = self.postprocess_frame(intermediate_tensor)
        
        return intermediate_frame
    
    def interpolate_video(self, video_path, output_path, interpolation_factor=2):
        """Interpolate frames in a video to increase frame rate"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps * interpolation_factor, (width, height))
        
        print(f"Processing video: {total_frames} frames, {fps} FPS")
        print(f"Output: {fps * interpolation_factor} FPS")
        
        ret, prev_frame = cap.read()
        if not ret:
            print("Error reading video")
            return
        
        # Write first frame
        out.write(prev_frame)
        
        frame_count = 1
        
        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            
            # Write current frame
            out.write(curr_frame)
            
            # Generate and write interpolated frame
            if frame_count < total_frames - 1:
                interpolated_frame = self.interpolate_frames(prev_frame, curr_frame)
                
                # Resize interpolated frame to match video dimensions
                interpolated_frame_resized = cv2.resize(interpolated_frame, (width, height))
                out.write(interpolated_frame_resized)
            
            prev_frame = curr_frame.copy()
            frame_count += 1
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        print(f"Video interpolation completed. Output saved to: {output_path}")
    
    def evaluate_interpolation(self, frame1, frame2, ground_truth):
        """Evaluate interpolation quality using SSIM and PSNR"""
        interpolated = self.interpolate_frames(frame1, frame2)
        
        # Convert to grayscale for SSIM calculation
        gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
        interp_gray = cv2.cvtColor(interpolated, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        ssim_score = ssim(gt_gray, interp_gray)
        psnr_score = psnr(gt_gray, interp_gray)
        
        return {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'interpolated_frame': interpolated
        }

def main():
    # Example usage
    interpolator = FrameInterpolator('best_model.pth')
    
    # Example: Interpolate between two frames
    if len(sys.argv) >= 3:
        frame1_path = sys.argv[1]
        frame2_path = sys.argv[2]
        
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is not None and frame2 is not None:
            interpolated = interpolator.interpolate_frames(frame1, frame2)
            
            # Save result
            output_path = 'interpolated_frame.jpg'
            cv2.imwrite(output_path, interpolated)
            print(f"Interpolated frame saved to: {output_path}")
            
            # Display frames
            cv2.imshow('Frame 1', frame1)
            cv2.imshow('Frame 2', frame2)
            cv2.imshow('Interpolated', interpolated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Could not read input frames")
    else:
        print("Usage: python inference.py <frame1_path> <frame2_path>")

if __name__ == '__main__':
    import sys
    main()
