# AI-Based Frame Interpolation

A deep learning system for generating smooth intermediate frames between two video frames using a UNet architecture. This project provides both a Python API and a web interface for frame interpolation.

## Features

- **Frame Interpolation**: Generate intermediate frames between two input frames
- **Video Interpolation**: Increase video frame rate by interpolating between existing frames
- **UNet Architecture**: Deep learning model based on UNet for high-quality interpolation
- **Web API**: FastAPI backend for easy integration
- **Modern Frontend**: Beautiful web interface with drag-and-drop functionality
- **Multiple Formats**: Support for various image and video formats
- **Quality Metrics**: SSIM and PSNR evaluation for interpolation quality

## Project Structure

```
AI-BASED-FRAME-INTERPOLATION/
├── model/                  # Deep learning model components
│   ├── unet.py           # UNet architecture implementation
│   ├── train.py          # Training script
│   └── inference.py      # Inference and evaluation
├── api/                   # FastAPI backend
│   └── app.py            # API endpoints
├── frontend/              # Web interface
│   ├── index.html        # Main HTML page
│   └── script.js         # Frontend JavaScript
├── data/                  # Dataset storage
├── main.py               # Command-line interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-BASED-FRAME-INTERPOLATION
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The project provides a comprehensive CLI through `main.py`:

#### Training
```bash
# Train the model with custom parameters
python main.py train --data-dir data/train --epochs 100 --batch-size 16

# Train with specific learning rate
python main.py train --data-dir data/train --epochs 200 --lr 0.0001
```

#### Inference
```bash
# Generate intermediate frame between two images
python main.py infer --frame1 frame1.jpg --frame2 frame2.jpg --output result.jpg

# Use specific model file
python main.py infer --frame1 frame1.jpg --frame2 frame2.jpg --output result.jpg --model my_model.pth
```

#### Video Interpolation
```bash
# Interpolate video with 2x frame rate
python main.py video --input video.mp4 --output interpolated.mp4 --factor 2

# Interpolate with 4x frame rate
python main.py video --input video.mp4 --output interpolated.mp4 --factor 4
```

#### Web API
```bash
# Start the API server
python main.py serve --host 0.0.0.0 --port 8000

# Development mode with auto-reload
python main.py serve --host 0.0.0.0 --port 8000 --reload
```

#### Model Information
```bash
# Show model details
python main.py info --model best_model.pth
```

### Web Interface

1. **Start the API server**
   ```bash
   python main.py serve
   ```

2. **Open the frontend**
   - Navigate to `frontend/index.html` in your browser
   - Or serve it using a local web server

3. **Use the interface**
   - Upload two frames for interpolation
   - Upload a video for frame rate increase
   - Download results directly

### API Endpoints

The FastAPI backend provides these endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `POST /interpolate-frames` - Interpolate between two frames
- `POST /interpolate-video` - Interpolate video frames
- `POST /evaluate-interpolation` - Evaluate interpolation quality
- `GET /model-info` - Model information

## Training

### Data Preparation

1. **Organize your data** in the following structure:
   ```
   data/
   ├── train/
   │   ├── video1/
   │   │   ├── frame_0000.jpg
   │   │   ├── frame_0001.jpg
   │   │   └── frame_0002.jpg
   │   └── video2/
   │       ├── frame_0000.jpg
   │       ├── frame_0001.jpg
   │       └── frame_0002.jpg
   └── val/
       └── ... (similar structure)
   ```

2. **Frame format**: Use sequential frame numbers (e.g., `frame_0000.jpg`, `frame_0001.jpg`)

3. **Image format**: JPG or PNG format recommended

### Training Process

1. **Start training**:
   ```bash
   python main.py train --data-dir data/train --epochs 100
   ```

2. **Monitor progress**: Training progress is displayed with loss metrics

3. **Model saving**: Best model is automatically saved as `best_model.pth`

## Model Architecture

The system uses a **UNet** architecture specifically designed for frame interpolation:

- **Input**: 6 channels (3 RGB channels × 2 frames)
- **Output**: 3 channels (RGB intermediate frame)
- **Architecture**: Encoder-decoder with skip connections
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate scheduling

## Performance

- **Training**: ~2-4 hours on GPU for 100 epochs
- **Inference**: Real-time for single frames, ~1-5 minutes for videos
- **Memory**: ~4-8GB GPU memory recommended for training
- **Quality**: SSIM > 0.8, PSNR > 25dB on test data

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in training
   - Use smaller input resolution
   - Enable gradient checkpointing

2. **Model not loading**
   - Check file path and permissions
   - Ensure model file exists and is not corrupted
   - Verify PyTorch version compatibility

3. **API connection errors**
   - Check if the server is running
   - Verify port configuration
   - Check firewall settings

### Getting Help

- Check the console output for error messages
- Verify all dependencies are installed correctly
- Ensure data format matches expected structure

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UNet architecture from the original paper
- FastAPI for the web framework
- PyTorch for deep learning capabilities
- OpenCV for computer vision operations

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{ai-frame-interpolation,
  title={AI-Based Frame Interpolation},
  author={Gaurav Daultani , Mohd Sahil , Nandini Bansal},
  year={2025},
  url={https://github.com/yourusername/ai-based-frame-interpolation}
}
```