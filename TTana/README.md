# TTana - AI Table Tennis Coach

## Technical Design Document

**Document Type:** Software / AI Technical Specification  
**Platform:** Windows Desktop  
**Language:** Python 3.11+  
**GUI:** PyQt6  
**LLM:** Qwen3.5-9B via LM Studio  
**Inference:** Local / Offline  
**Video Input:** Single-camera video  

---

## Overview

TTana is a desktop application for table tennis technique analysis using single-camera videos. It compares user videos with professional reference videos using:

- Pose Estimation (RTMPose via MMPose)
- Human Tracking (ByteTrack)
- Temporal Motion Analysis
- Biomechanical Feature Extraction
- Stroke Segmentation & Phase Detection
- Dynamic Time Warping (DTW)
- Motion Comparison
- Root-Cause Analysis
- Local LLM Reasoning (Qwen3.5-9B)

---

## Architecture

```
┌───────────────────────────────────────────────────────┐
│                    PyQt6 Application                  │
│   Dashboard | Video | Comparison | Biomechanics | Chat│
└──────────────────────────┬────────────────────────────┘
                           │
                           ▼
┌───────────────────────────────────────────────────────┐
│                  Application Core                     │
│   Analysis Orchestrator | Session Manager | Cache     │
└──────────────────────────┬────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌───────────────────────┐     ┌────────────────────────┐
│     Vision Engine     │     │     LLM Coach Engine   │
│   OpenCV | RTMPose    │     │   LM Studio | Qwen3.5  │
│   Tracking | Smoothing│     │   Prompt | Context     │
└───────────┬───────────┘     └────────────┬───────────┘
            │                              │
            ▼                              │
┌───────────────────────────┐              │
│   Biomechanics Engine     │◄─────────────┘
│   Angles | Velocity | DTW │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│      Analysis Context     │
│   Structured JSON Output  │
└───────────────────────────┘
```

---

## Prerequisites

### System Requirements

- **OS:** Windows 10/11 (64-bit)
- **Python:** 3.11 or higher
- **GPU:** NVIDIA GPU with CUDA support (recommended: RTX 2060 Super 8GB or better)
- **RAM:** 16GB minimum, 32GB recommended
- **Storage:** 10GB free space for models and cache

### Software Dependencies

1. **Python 3.11+**
   - Download from: https://www.python.org/downloads/
   - Ensure "Add Python to PATH" is checked during installation

2. **CUDA Toolkit** (for GPU acceleration)
   - Version: 11.8 or 12.x
   - Download from: https://developer.nvidia.com/cuda-toolkit-archive

3. **FFmpeg**
   - Download from: https://ffmpeg.org/download.html
   - Add to system PATH

4. **LM Studio** (for local LLM)
   - Download from: https://lmstudio.ai/
   - Install Qwen3.5-9B model within LM Studio
   - Enable local API server (default: http://localhost:1234)

---

## Installation

### Step 1: Clone or Navigate to Project Directory

```bash
cd TTana
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install MMPose (Special Installation)

MMPose requires special installation:

```bash
# Install MMCV
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# Install MMPose
pip install mmpose
```

### Step 6: Download Models

Models will be automatically downloaded on first run, or manually place them in the `models/` directory:

- RTMPose model weights
- RTMDet model weights

---

## Configuration

Edit `config.yaml` to customize:

```yaml
pose:
  confidence_threshold: 0.5
  model_type: "rtmpose"

tracking:
  method: "bytetrack"
  confidence_threshold: 0.5

analysis:
  smoothing_window: 7
  dtw_window: 20

llm:
  endpoint: "http://localhost:1234/v1"
  model: "qwen3.5-9b"
  temperature: 0.2
  max_tokens: 4096

gpu:
  use_cuda: true
  device: "cuda:0"
```

---

## Usage

### Starting the Application

```bash
python app/main.py
```

### Workflow

1. **Launch Application**
   - Run the main script
   - PyQt6 GUI will open

2. **Load Videos**
   - Click "Select User Video" to load your table tennis video
   - Click "Select Reference Video" to load professional video

3. **Start Analysis**
   - Click "Analyze" button
   - Wait for pipeline completion:
     - Video decoding
     - Person detection
     - Pose estimation
     - Tracking & smoothing
     - Stroke detection
     - Phase segmentation
     - Feature extraction
     - DTW alignment
     - Comparison & diagnosis
     - LLM report generation

4. **View Results**
   - **Dashboard:** Overview of analysis
   - **Video Player:** Side-by-side comparison with skeleton overlay
   - **Biomechanics:** Joint angles, velocities, timing data
   - **Comparison:** Feature-by-feature comparison
   - **Coach Report:** AI-generated diagnosis and recommendations

5. **Chat with AI Coach**
   - Ask questions about your technique
   - Examples:
     - "What's wrong with my backswing?"
     - "How can I improve my timing?"
     - "Show me my strongest stroke"
   - Click "View Evidence" to see video segments related to diagnoses

---

## Project Structure

```
TTana/
│
├── app/
│   ├── main.py                  # Application entry point
│   ├── core/
│   │   ├── pipeline.py          # Analysis orchestrator
│   │   ├── session.py           # Session management
│   │   └── config.py            # Configuration loader
│   ├── vision/
│   │   ├── detector.py          # Person detection (RTMDet)
│   │   ├── pose.py              # Pose estimation (RTMPose)
│   │   ├── tracker.py           # Tracking (ByteTrack)
│   │   └── smoothing.py         # Temporal smoothing
│   ├── biomechanics/
│   │   ├── angles.py            # Joint angle calculations
│   │   ├── kinematics.py        # Velocity & acceleration
│   │   ├── features.py          # Feature extraction
│   │   └── normalization.py     # Coordinate normalization
│   ├── strokes/
│   │   ├── detector.py          # Stroke detection
│   │   └── phases.py            # Phase segmentation
│   ├── alignment/
│   │   ├── dtw.py               # Dynamic Time Warping
│   │   └── matcher.py           # Feature matching
│   ├── diagnosis/
│   │   ├── engine.py            # Diagnosis engine
│   │   ├── root_cause.py        # Root cause analysis
│   │   └── confidence.py        # Confidence scoring
│   ├── llm/
│   │   ├── client.py            # LM Studio client
│   │   ├── prompts.py           # Prompt templates
│   │   ├── context.py           # Context builder
│   │   └── validator.py         # JSON validation
│   ├── database/
│   │   └── repository.py        # SQLite repository
│   └── ui/
│       ├── main_window.py       # Main window
│       ├── dashboard.py         # Dashboard view
│       ├── video.py             # Video player
│       ├── comparison.py        # Comparison view
│       ├── biomechanics.py      # Biomechanics view
│       └── coach.py             # Coach chat interface
│
├── models/                      # Model weights
├── cache/                       # Cached analysis results
├── sessions/                    # Session data
├── tests/                       # Unit tests
├── docs/                        # Documentation
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Pipeline Stages

1. **Video Loading & Validation**
   - Check video format
   - Extract metadata
   - Generate hash for caching

2. **Frame Extraction**
   - Decode video using FFmpeg/OpenCV
   - Normalize frame rate and resolution

3. **Person Detection**
   - Detect persons using RTMDet
   - Select target player (largest/highest confidence)

4. **Pose Estimation**
   - Estimate keypoints using RTMPose
   - Filter by confidence threshold

5. **Tracking**
   - Track person across frames using ByteTrack
   - Handle occlusions and ID switches

6. **Temporal Smoothing**
   - Apply Savitzky-Golay filter
   - Remove outliers
   - Interpolate missing keypoints

7. **Stroke Detection**
   - Detect strokes using motion energy
   - Allow manual override

8. **Phase Segmentation**
   - Segment strokes into phases:
     - Preparation
     - Backswing
     - Acceleration
     - Contact (proxy)
     - Follow-through
     - Recovery

9. **Feature Extraction**
   - Joint angles
   - Segment orientations
   - Angular velocity & acceleration
   - Linear velocity
   - Motion chain timing

10. **Normalization**
    - Normalize coordinates
    - Scale normalization based on body size

11. **DTW Alignment**
    - Align user and reference sequences
    - Apply feature weighting

12. **Comparison**
    - Feature-level comparison
    - Calculate similarity scores

13. **Diagnosis**
    - Identify issues
    - Perform root cause analysis
    - Assign confidence scores

14. **LLM Report Generation**
    - Build analysis context
    - Generate structured report via Qwen3.5-9B

15. **Visualization**
    - Render skeleton overlays
    - Display difference vectors
    - Show timeline with phases

---

## API Endpoints

### Internal Python API

```python
from app.core.pipeline import AnalysisPipeline
from app.core.session import SessionManager

# Create session
session = SessionManager.create_session(
    user_video_path="path/to/user.mp4",
    reference_video_path="path/to/pro.mp4"
)

# Run analysis
pipeline = AnalysisPipeline(session)
result = pipeline.run()

# Generate coach report
from app.llm.client import LLMClient
report = LLMClient.generate_report(result)

# Chat with coach
answer = LLMClient.chat(
    session=session,
    message="What's my main issue?"
)
```

---

## LLM Integration

### LM Studio Setup

1. Download and install LM Studio
2. Download Qwen3.5-9B model
3. Start local server:
   - Go to "Local Server" tab
   - Click "Start Server"
   - Default endpoint: `http://localhost:1234/v1`

### System Prompt

The LLM operates as:
- Elite Table Tennis Coach
- Biomechanics Analyst
- Evidence-based Training Advisor

### Rules

1. Use only provided Analysis Context data
2. Do not invent numbers
3. Distinguish between measurement and inference
4. Separate root cause from symptoms
5. Prioritize most important correction
6. Provide measurable training recommendations

---

## Caching Strategy

Analysis results are cached based on video hash (SHA256):

```
cache/
 └── <video_hash>/
      ├── metadata.json
      ├── poses.npz
      ├── features.npz
      └── strokes.json
```

If video hasn't changed, pose estimation is skipped.

---

## Database Schema

SQLite database with tables:

- `players` - Player information
- `sessions` - Analysis sessions
- `videos` - Video metadata
- `strokes` - Detected strokes
- `features` - Biomechanical features
- `comparisons` - Comparison results
- `diagnoses` - Diagnosis data
- `coach_reports` - LLM-generated reports
- `chat_messages` - Chat history

---

## Error Handling

The system handles:

- Invalid video formats
- No person detected
- Multiple people in frame
- Low pose confidence
- Insufficient frames
- Poor visibility
- Unsupported codecs
- LM Studio unavailability
- Invalid LLM JSON responses
- GPU memory errors

---

## Security & Privacy

- **All processing is local**
- No internet connection required after setup
- No data sent to external services
- Videos remain on user's machine
- LLM runs locally via LM Studio

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce batch size in config
- Close other GPU applications
- Use smaller model variants

**2. LM Studio connection failed**
- Ensure LM Studio server is running
- Check endpoint URL in config
- Verify firewall settings

**3. No person detected**
- Ensure good lighting
- Use videos with clear player visibility
- Adjust confidence threshold in config

**4. Slow performance**
- Enable GPU acceleration
- Reduce video resolution
- Use cached results when available

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

Follow PEP 8 guidelines. Use type hints where possible.

### Adding New Features

1. Create feature branch
2. Implement changes
3. Add tests
4. Update documentation
5. Submit pull request

---

## License

[Specify license here]

---

## Support

For issues and questions, please refer to the documentation in the `docs/` folder or contact the development team.

---

## Version History

- **v1.0.0** - Initial release
  - Single-camera video analysis
  - Pose estimation with RTMPose
  - Biomechanical feature extraction
  - DTW-based comparison
  - Local LLM coaching with Qwen3.5-9B
  - PyQt6 GUI
