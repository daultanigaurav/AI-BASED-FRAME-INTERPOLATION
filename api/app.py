from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
import tempfile
import uuid
from typing import List
import sys
import os

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from inference import FrameInterpolator

app = FastAPI(
    title="AI-Based Frame Interpolation API",
    description="API for interpolating frames using deep learning models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the frame interpolator
try:
    interpolator = FrameInterpolator('best_model.pth')
    print("Frame interpolator initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize frame interpolator: {e}")
    interpolator = None

@app.get("/")
async def root():
    return {"message": "AI-Based Frame Interpolation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "interpolator_ready": interpolator is not None}

@app.post("/interpolate-frames")
async def interpolate_frames(
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...)
):
    """
    Interpolate between two frames to generate an intermediate frame
    """
    if interpolator is None:
        raise HTTPException(status_code=500, detail="Frame interpolator not available")
    
    # Validate file types
    if not frame1.content_type.startswith('image/') or not frame2.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Both files must be images")
    
    try:
        # Read and decode images
        frame1_content = await frame1.read()
        frame2_content = await frame2.read()
        
        # Convert to numpy arrays
        frame1_np = np.frombuffer(frame1_content, np.uint8)
        frame2_np = np.frombuffer(frame2_content, np.uint8)
        
        frame1_img = cv2.imdecode(frame1_np, cv2.IMREAD_COLOR)
        frame2_img = cv2.imdecode(frame2_np, cv2.IMREAD_COLOR)
        
        if frame1_img is None or frame2_img is None:
            raise HTTPException(status_code=400, detail="Could not decode image files")
        
        # Generate interpolated frame
        interpolated_frame = interpolator.interpolate_frames(frame1_img, frame2_img)
        
        # Save interpolated frame to temporary file
        temp_dir = tempfile.gettempdir()
        output_filename = f"interpolated_{uuid.uuid4()}.jpg"
        output_path = os.path.join(temp_dir, output_filename)
        
        cv2.imwrite(output_path, interpolated_frame)
        
        # Return the file
        return FileResponse(
            output_path,
            media_type="image/jpeg",
            filename=output_filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")

@app.post("/interpolate-video")
async def interpolate_video(
    video: UploadFile = File(...),
    interpolation_factor: int = 2
):
    """
    Interpolate frames in a video to increase frame rate
    """
    if interpolator is None:
        raise HTTPException(status_code=500, detail="Frame interpolator not available")
    
    # Validate file type
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded video to temporary file
        temp_dir = tempfile.gettempdir()
        input_filename = f"input_{uuid.uuid4()}.mp4"
        input_path = os.path.join(temp_dir, input_filename)
        
        with open(input_path, "wb") as buffer:
            content = await video.read()
            buffer.write(content)
        
        # Generate output filename
        output_filename = f"interpolated_{uuid.uuid4()}.mp4"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Process video
        interpolator.interpolate_video(input_path, output_path, interpolation_factor)
        
        # Clean up input file
        os.remove(input_path)
        
        # Return the processed video
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=output_filename
        )
        
    except Exception as e:
        # Clean up files on error
        if 'input_path' in locals() and os.path.exists(input_path):
            os.remove(input_path)
        if 'output_path' in locals() and os.path.exists(output_path):
            os.remove(output_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/evaluate-interpolation")
async def evaluate_interpolation(
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...),
    ground_truth: UploadFile = File(...)
):
    """
    Evaluate interpolation quality using SSIM and PSNR metrics
    """
    if interpolator is None:
        raise HTTPException(status_code=500, detail="Frame interpolator not available")
    
    try:
        # Read and decode images
        frame1_content = await frame1.read()
        frame2_content = await frame2.read()
        gt_content = await ground_truth.read()
        
        frame1_img = cv2.imdecode(np.frombuffer(frame1_content, np.uint8), cv2.IMREAD_COLOR)
        frame2_img = cv2.imdecode(np.frombuffer(frame2_content, np.uint8), cv2.IMREAD_COLOR)
        gt_img = cv2.imdecode(np.frombuffer(gt_content, np.uint8), cv2.IMREAD_COLOR)
        
        if any(img is None for img in [frame1_img, frame2_img, gt_img]):
            raise HTTPException(status_code=400, detail="Could not decode image files")
        
        # Evaluate interpolation
        results = interpolator.evaluate_interpolation(frame1_img, frame2_img, gt_img)
        
        # Save interpolated frame to temporary file
        temp_dir = tempfile.gettempdir()
        output_filename = f"evaluated_{uuid.uuid4()}.jpg"
        output_path = os.path.join(temp_dir, output_filename)
        cv2.imwrite(output_path, results['interpolated_frame'])
        
        return {
            "ssim": results['ssim'],
            "psnr": results['psnr'],
            "interpolated_frame": FileResponse(
                output_path,
                media_type="image/jpeg",
                filename=output_filename
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating interpolation: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """
    Get information about the loaded model
    """
    if interpolator is None:
        raise HTTPException(status_code=500, detail="Frame interpolator not available")
    
    return {
        "model_type": "FrameInterpolationUNet",
        "input_channels": 6,  # 3 channels per frame
        "output_channels": 3,
        "device": str(interpolator.device),
        "model_loaded": interpolator.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
