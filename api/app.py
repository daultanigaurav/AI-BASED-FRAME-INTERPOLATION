from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import uuid
from typing import Optional

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

app = FastAPI(
    title="AI Frame Interpolation API",
    description="API for generating intermediate frames between two input images using U-Net",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("temp_outputs")
MODEL_PATH = "best_model.pth"  # Default model path
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
    except Exception as e:
        print(f"Warning: Could not clean up temp files: {e}")

def validate_image_file(file: UploadFile) -> bool:
    """Validate uploaded image file"""
    if not file.content_type or not file.content_type.startswith('image/'):
        return False
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return False
    
    return True

def run_inference(frame1_path: str, frame2_path: str, num_intermediate: int = 3, fps: int = 30) -> str:
    """
    Run the inference pipeline using the inference.py script
    
    Args:
        frame1_path: Path to first frame
        frame2_path: Path to second frame
        num_intermediate: Number of intermediate frames to generate
        fps: Frames per second for output video
    
    Returns:
        Path to generated video file
    """
    try:
        # Construct the command
        inference_script = Path(__file__).parent.parent / "model" / "inference.py"
        
        cmd = [
            sys.executable,  # Use current Python interpreter
            str(inference_script),
            "--frame1", frame1_path,
            "--frame2", frame2_path,
            "--num-intermediate", str(num_intermediate),
            "--fps", str(fps),
            "--output", "output.png"  # This will be converted to video.mp4
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the inference script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=OUTPUT_DIR,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Error running inference: {result.stderr}")
            raise RuntimeError(f"Inference failed: {result.stderr}")
        
        print(f"Inference output: {result.stdout}")
        
        # Check for generated video file
        video_path = OUTPUT_DIR / "video.mp4"
        if not video_path.exists():
            raise FileNotFoundError("Video file was not generated")
        
        return str(video_path)
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("Inference timed out after 5 minutes")
    except Exception as e:
        raise RuntimeError(f"Failed to run inference: {str(e)}")

@app.post("/interpolate")
async def interpolate_frames(
    frame1: UploadFile = File(..., description="First input frame"),
    frame2: UploadFile = File(..., description="Second input frame"),
    num_intermediate: int = Form(3, description="Number of intermediate frames to generate (1-10)"),
    fps: int = Form(30, description="Frames per second for output video (10-60)")
):
    """
    Generate intermediate frames between two input images and return a video
    
    - **frame1**: First input frame (JPG, PNG, BMP, TIFF)
    - **frame2**: Second input frame (JPG, PNG, BMP, TIFF)
    - **num_intermediate**: Number of intermediate frames (1-10)
    - **fps**: Output video frame rate (10-60)
    
    Returns a video file (MP4) containing the original frames and generated intermediate frames.
    """
    
    # Validate parameters
    if num_intermediate < 1 or num_intermediate > 10:
        raise HTTPException(status_code=400, detail="num_intermediate must be between 1 and 10")
    
    if fps < 10 or fps > 60:
        raise HTTPException(status_code=400, detail="fps must be between 10 and 60")
    
    # Validate uploaded files
    if not frame1.filename or not frame2.filename:
        raise HTTPException(status_code=400, detail="Both frame files must be provided")
    
    if not validate_image_file(frame1):
        raise HTTPException(status_code=400, detail="frame1 must be a valid image file (JPG, PNG, BMP, TIFF)")
    
    if not validate_image_file(frame2):
        raise HTTPException(status_code=400, detail="frame2 must be a valid image file (JPG, PNG, BMP, TIFF)")
    
    # Create unique session ID for this request
    session_id = str(uuid.uuid4())
    session_upload_dir = UPLOAD_DIR / session_id
    session_output_dir = OUTPUT_DIR / session_id
    
    try:
        # Create session directories
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        session_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        frame1_path = session_upload_dir / f"frame1{Path(frame1.filename).suffix}"
        frame2_path = session_upload_dir / f"frame2{Path(frame2.filename).suffix}"
        
        with open(frame1_path, "wb") as f:
            shutil.copyfileobj(frame1.file, f)
        
        with open(frame2_path, "wb") as f:
            shutil.copyfileobj(frame2.file, f)
        
        print(f"Saved uploaded files: {frame1_path}, {frame2_path}")
        
        # Run inference
        video_path = run_inference(
            str(frame1_path), 
            str(frame2_path), 
            num_intermediate, 
            fps
        )
        
        # Return video file
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=f"interpolated_frames_{num_intermediate}inter_{fps}fps.mp4"
        )
        
    except Exception as e:
        print(f"Error during interpolation: {e}")
        raise HTTPException(status_code=500, detail=f"Interpolation failed: {str(e)}")
    
    finally:
        # Clean up session files
        try:
            if session_upload_dir.exists():
                shutil.rmtree(session_upload_dir)
            if session_output_dir.exists():
                shutil.rmtree(session_output_dir)
        except Exception as e:
            print(f"Warning: Could not clean up session files: {e}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "AI Frame Interpolation API",
        "version": "1.0.0",
        "endpoints": {
            "/interpolate": "POST - Generate intermediate frames between two images",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.on_event("startup")
async def startup_event():
    """Initialize API on startup"""
    print("Starting AI Frame Interpolation API...")
    print(f"Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"Warning: Model file {MODEL_PATH} not found. Training required before inference.")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down AI Frame Interpolation API...")
    cleanup_temp_files()

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
