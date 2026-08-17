# AI Table Tennis Coach - Technical Requirements

## System Requirements

### Hardware
- **CPU**: Intel Core i7 or AMD Ryzen 7 (or better)
- **GPU**: NVIDIA RTX 2060 Super 8GB or better (CUDA support required)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space for models and cache
- **OS**: Windows 10/11 (64-bit)

### Software
- **Python**: 3.11 or higher
- **CUDA Toolkit**: 11.8 or 12.x
- **cuDNN**: Compatible with CUDA version
- **FFmpeg**: Latest version for video processing
- **LM Studio**: For running Qwen3.5-9B locally

## Installation Steps

### 1. Install Python 3.11+
Download from [python.org](https://www.python.org/downloads/)

### 2. Install CUDA Toolkit
Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

### 3. Install FFmpeg
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add to system PATH

### 4. Install LM Studio
- Download from [lmstudio.ai](https://lmstudio.ai/)
- Install Qwen3.5-9B model
- Start local server on `http://localhost:1234`

### 5. Clone and Setup Project
```bash
cd TTana
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 6. Download Models
Models will be automatically downloaded on first run, or manually place in `models/` directory.

### 7. Configure
Edit `config.yaml` to adjust settings for your hardware.

### 8. Run Application
```bash
python app/main.py
```

## Usage Workflow

1. **Launch Application**
   - Run `python app/main.py`
   - Wait for UI to load

2. **Load Videos**
   - Click "Select User Video" to load your table tennis video
   - Click "Select Reference Video" to load professional video
   - Supported formats: MP4, AVI, MOV, MKV

3. **Configure Analysis**
   - Set handedness (Left/Right/Auto)
   - Adjust confidence thresholds if needed
   - Select stroke types to analyze

4. **Run Analysis**
   - Click "Start Analysis"
   - Wait for pipeline to complete (may take several minutes)
   - Progress shown in status bar

5. **Review Results**
   - **Dashboard**: Overview of analysis
   - **Video Player**: Side-by-side comparison with skeleton overlay
   - **Biomechanics**: Detailed angle, velocity, and timing data
   - **Coach Report**: AI-generated diagnosis and recommendations
   - **Chat**: Ask questions about your technique

6. **Save Session**
   - Sessions are automatically saved
   - Access previous sessions from the dashboard

## Troubleshooting

### Common Issues

**No person detected**
- Ensure good lighting in video
- Player should be clearly visible
- Adjust confidence threshold in config

**LM Studio connection failed**
- Ensure LM Studio is running
- Check server is started on port 1234
- Verify Qwen3.5-9B model is loaded

**Slow performance**
- Close other GPU-intensive applications
- Reduce video resolution
- Adjust batch size in config

**CUDA out of memory**
- Close other applications using GPU
- Reduce model size or batch size
- Consider using quantized models

## Support

For issues and feature requests, please check documentation in `docs/` folder.
