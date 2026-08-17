# TTana - AI Table Tennis Coach

## Single-Camera Video Analysis System

**Version:** 1.0.0  
**Platform:** Windows Desktop  
**Language:** Python 3.11+  
**GUI:** PyQt6  
**LLM:** Qwen3.5-9B via LM Studio (Local)  

---

## Overview

TTana is a professional desktop application for table tennis technique analysis using single-camera video. It compares user videos with professional references using advanced computer vision, biomechanical analysis, and local AI coaching.

## Key Features

### 🎯 Core Capabilities
- **Pose Estimation**: RTMPose via MMPose for accurate skeleton tracking
- **Motion Tracking**: ByteTrack with Kalman filtering for smooth trajectories
- **Biomechanical Analysis**: Joint angles, velocities, accelerations, timing
- **Stroke Segmentation**: Automatic detection of preparation, backswing, acceleration, contact, follow-through, recovery
- **Temporal Alignment**: Dynamic Time Warping (DTW) for accurate comparison
- **Root Cause Analysis**: Intelligent diagnosis of technique issues
- **AI Coach Chat**: Conversational interface with Qwen3.5-9B for personalized feedback

### 🔒 Privacy & Security
- **100% Local Processing**: No internet required after setup
- **No Cloud APIs**: All inference runs on your machine
- **Video Privacy**: Your footage never leaves your device
- **Local LLM**: Qwen3.5-9B runs via LM Studio locally

### 🖥️ User Interface
- **Dashboard**: Session overview and quick stats
- **Video Comparison**: Side-by-side playback with skeleton overlay
- **Biomechanics Panel**: Detailed angle and velocity graphs
- **Coach Report**: AI-generated diagnosis and training plan
- **Chat Interface**: Ask questions about your technique

---

## Quick Start

### Prerequisites
- Windows 10/11 (64-bit)
- NVIDIA GPU with 8GB+ VRAM (RTX 2060 Super or better)
- Python 3.11+
- CUDA Toolkit 11.8 or 12.x
- FFmpeg
- LM Studio with Qwen3.5-9B model

### Installation

```bash
# Clone or navigate to project
cd TTana

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
python app/main.py
```

### First-Time Setup

1. **Install LM Studio**: Download from [lmstudio.ai](https://lmstudio.ai/)
2. **Download Qwen3.5-9B**: In LM Studio, search and download the model
3. **Start Local Server**: In LM Studio, go to Local Server tab and start server on `http://localhost:1234`
4. **Run TTana**: Execute `python app/main.py`

---

## Usage Workflow

### 1. Load Videos
- Click **"Select User Video"** to load your table tennis footage
- Click **"Select Reference Video"** to load a professional example
- Supported formats: MP4, AVI, MOV, MKV

### 2. Configure Analysis
- Set player handedness (Left/Right/Auto)
- Adjust confidence thresholds if needed
- Select stroke types to analyze

### 3. Run Analysis
- Click **"Start Analysis"**
- Wait for the pipeline to complete:
  - Video decoding and frame extraction
  - Person detection and pose estimation
  - Tracking and temporal smoothing
  - Stroke segmentation and phase detection
  - Biomechanical feature extraction
  - DTW alignment and comparison
  - Root cause analysis
  - AI coach report generation

### 4. Review Results
- **Dashboard**: Overall similarity score and key metrics
- **Video Player**: Synchronized playback with skeleton overlays
- **Biomechanics**: Interactive graphs of angles, velocities, timing
- **Coach Report**: Structured diagnosis with evidence
- **Chat**: Ask specific questions about your technique

### 5. Save & Export
- Sessions are automatically saved to `sessions/`
- Export reports as PDF or JSON
- Access previous sessions from dashboard

---

## Architecture

```
┌─────────────────────────────────────────┐
│           PyQt6 Application             │
│  Dashboard | Video | Comparison | Chat  │
└────────────────────┬────────────────────┘
                     │
┌────────────────────▼────────────────────┐
│         Application Core                │
│  Pipeline Orchestrator | Session Manager│
└──────────┬──────────────────┬───────────┘
           │                  │
┌──────────▼──────────┐ ┌────▼────────────┐
│   Vision Engine     │ │  LLM Engine     │
│  OpenCV | RTMPose   │ │ LM Studio Client│
│  ByteTrack | Filter │ │ Qwen3.5-9B      │
└──────────┬──────────┘ │ Prompt Manager  │
           │            └────────┬────────┘
┌──────────▼──────────┐          │
│ Biomechanics Engine │◄─────────┘
│ Angles | Velocity   │
│ Acceleration | DTW  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Diagnosis Engine   │
│ Root Cause Analysis │
│ Confidence Scoring  │
└─────────────────────┘
```

---

## Technology Stack

### Core
- **Python 3.11+**: Main programming language
- **PyQt6**: Desktop GUI framework
- **OpenCV**: Video I/O and frame processing
- **FFmpeg**: Video decoding and normalization
- **NumPy**: Numerical computations
- **SciPy**: Signal processing and smoothing
- **Pydantic**: Data validation
- **SQLite**: Session and data storage

### Vision
- **PyTorch**: Deep learning framework
- **CUDA**: GPU acceleration
- **MMPose**: Pose estimation framework
- **RTMPose**: State-of-the-art pose model
- **RTMDet**: Person detection
- **ByteTrack**: Multi-object tracking
- **Savitzky-Golay**: Temporal smoothing

### Analysis
- **dtaidistance**: Dynamic Time Warping
- **Custom DTW**: Multivariate weighted alignment
- **Signal Processing**: Velocity, acceleration, angular features

### AI Coach
- **LM Studio**: Local LLM server
- **Qwen3.5-9B**: 9B parameter language model
- **Custom Prompts**: Coaching-specific system prompts
- **Context Retrieval**: Smart context selection for chat

---

## Project Structure

```
TTana/
│
├── app/                          # Main application code
│   ├── main.py                   # Entry point
│   │
│   ├── core/                     # Core modules
│   │   ├── pipeline.py           # Analysis orchestrator
│   │   ├── session.py            # Session management
│   │   ├── config.py             # Configuration loader
│   │   └── cache.py              # Caching system
│   │
│   ├── vision/                   # Computer vision
│   │   ├── detector.py           # Person detection
│   │   ├── pose.py               # Pose estimation
│   │   ├── tracker.py            # Multi-object tracking
│   │   └── smoothing.py          # Temporal filtering
│   │
│   ├── biomechanics/             # Biomechanical analysis
│   │   ├── angles.py             # Joint angle calculation
│   │   ├── kinematics.py         # Velocity & acceleration
│   │   ├── features.py           # Feature extraction
│   │   └── normalization.py      # Coordinate normalization
│   │
│   ├── strokes/                  # Stroke analysis
│   │   ├── detector.py           # Stroke detection
│   │   └── phases.py             # Phase segmentation
│   │
│   ├── alignment/                # Temporal alignment
│   │   ├── dtw.py                # Dynamic Time Warping
│   │   └── matcher.py            # Feature matching
│   │
│   ├── diagnosis/                # Diagnosis engine
│   │   ├── engine.py             # Diagnostic reasoning
│   │   ├── root_cause.py         # Root cause analysis
│   │   └── confidence.py         # Confidence scoring
│   │
│   ├── llm/                      # LLM integration
│   │   ├── client.py             # LM Studio API client
│   │   ├── prompts.py            # System prompts
│   │   ├── context.py            # Context builder
│   │   └── validator.py          # Output validation
│   │
│   ├── database/                 # Data persistence
│   │   └── repository.py         # SQLite operations
│   │
│   └── ui/                       # User interface
│       ├── main_window.py        # Main window
│       ├── dashboard.py          # Dashboard view
│       ├── video_player.py       # Video playback
│       ├── comparison.py         # Side-by-side view
│       ├── biomechanics.py       # Biomechanics panel
│       ├── coach.py              # Coach report view
│       └── chat.py               # Chat interface
│
├── models/                       # Model weights
├── cache/                        # Cached analysis results
├── sessions/                     # Session data
├── tests/                        # Unit and integration tests
├── docs/                         # Documentation
│   ├── INSTALL_FA.md             # Persian installation guide
│   └── TECHNICAL_REQUIREMENTS.md # Technical specs
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Configuration

Edit `config.yaml` to customize behavior:

```yaml
# Pose estimation settings
pose:
  confidence_threshold: 0.5
  model: "rtmpose-l"
  device: "cuda:0"

# Tracking settings
tracking:
  track_confidence: 0.6
  smoothing_window: 7

# DTW settings
dtw:
  window: 20
  use_weights: true

# LLM settings
llm:
  endpoint: "http://localhost:1234/v1"
  model: "qwen3.5-9b"
  temperature: 0.2
  max_tokens: 4096
  timeout: 120

# Feature weights for comparison
feature_weights:
  shoulder_timing: 1.5
  hip_rotation: 1.4
  elbow_trajectory: 1.3
  wrist_angle: 1.0
  knee_angle: 0.9

# GPU settings
gpu:
  use_cuda: true
  device: "cuda:0"
  batch_size: 8
```

---

## Analysis Pipeline

The complete analysis flow:

1. **Video Loading**: Load and validate user and reference videos
2. **Frame Extraction**: Decode videos using FFmpeg/OpenCV
3. **Person Detection**: Detect players using RTMDet
4. **Pose Estimation**: Extract keypoints using RTMPose
5. **Tracking**: Associate poses across frames with ByteTrack
6. **Smoothing**: Apply Savitzky-Golay filter to reduce jitter
7. **Stroke Detection**: Identify stroke boundaries from motion energy
8. **Phase Segmentation**: Divide strokes into preparation, backswing, acceleration, contact, follow-through, recovery
9. **Feature Extraction**: Calculate angles, velocities, accelerations, timing
10. **Normalization**: Normalize coordinates and timelines
11. **DTW Alignment**: Align user and reference strokes temporally
12. **Comparison**: Compute feature-level differences
13. **Diagnosis**: Identify root causes and priority issues
14. **Report Generation**: Create structured coach report via LLM
15. **Chat Enablement**: Allow conversational queries about analysis

---

## Biomechanical Features

### Joint Angles
- Elbow flexion/extension
- Shoulder abduction/adduction
- Hip rotation
- Knee flexion
- Trunk angle

### Kinematics
- Linear velocity (wrist, elbow, shoulder, hip)
- Angular velocity (joint rotation speeds)
- Acceleration profiles
- Motion energy curves

### Timing
- Phase durations
- Peak velocity timing
- Joint sequencing (kinematic chain)
- Relative timing between joints

### Spatial
- Trajectory paths
- Body-relative positions
- Scale-normalized coordinates

---

## AI Coach System

### System Prompt
The LLM operates as:
- Elite Table Tennis Coach
- Biomechanics Analyst
- Evidence-based Training Advisor

### Rules
1. Use only data from Analysis Context
2. Never invent numbers
3. Distinguish measurement from inference
4. Separate root cause from symptoms
5. Prioritize most impactful correction
6. Provide measurable training recommendations

### Output Structure
```json
{
  "summary": "Overall assessment",
  "primary_issue": "Main technical problem",
  "root_cause": "Underlying cause",
  "evidence": ["frame ranges", "measurements"],
  "secondary_issues": ["additional problems"],
  "strengths": ["what user does well"],
  "corrections": ["specific fixes"],
  "training_plan": ["drills and exercises"],
  "next_session_goal": "focus for next practice"
}
```

### Chat Context Retrieval
When user asks questions, the system:
1. Detects intent (timing, wrist, hip, etc.)
2. Retrieves relevant analysis sections
3. Builds focused context
4. Sends to LLM with conversation history
5. Returns evidence-linked answer

---

## Caching System

Analysis results are cached by video hash (SHA256):

```
cache/
 └── <video_hash>/
      ├── metadata.json       # Video info
      ├── poses.npz           # Keypoint data
      ├── features.npz        # Biomechanical features
      └── strokes.json        # Stroke segments
```

If the same video is analyzed again, cached results are used instead of re-running pose estimation.

---

## Database Schema

SQLite database stores:

```sql
-- Players table
players (id, name, handedness, created_at)

-- Sessions table
sessions (id, player_id, user_video, ref_video, status, created_at)

-- Videos table
videos (id, session_id, type, path, hash, duration, fps)

-- Strokes table
strokes (id, session_id, stroke_type, start_frame, end_frame, phase_data)

-- Features table
features (id, stroke_id, feature_name, values, timestamps)

-- Comparisons table
comparisons (id, session_id, dtw_distance, similarity_score, feature_diffs)

-- Diagnoses table
diagnoses (id, session_id, primary_issue, root_cause, confidence)

-- Coach Reports table
coach_reports (id, session_id, report_json, generated_at)

-- Chat Messages table
chat_messages (id, session_id, role, content, timestamp)
```

---

## Error Handling

The system handles:

- Invalid video formats
- No person detected in frame
- Multiple people (selects target)
- Low pose confidence frames
- Insufficient frames for analysis
- Poor visibility conditions
- Unsupported video codecs
- LM Studio connection failures
- Invalid LLM JSON responses
- GPU memory errors

Graceful degradation ensures partial results when possible.

---

## Performance Optimization

### GPU Strategy
- Pose model runs on CUDA
- OpenCV uses GPU acceleration when available
- NumPy/SciPy operations on CPU
- DTW computation on CPU (memory-intensive)
- Qwen runs in separate LM Studio process

### Memory Management
- Batch processing for large videos
- Frame subsampling for initial analysis
- Lazy loading of video frames
- Efficient caching with npz format
- Model unloading between sessions

### Threading
- UI thread never blocked
- Background workers for heavy computation
- QThreadPool for parallel tasks
- Progress signals to update UI
- Cancelable operations

---

## Testing

Run tests with pytest:

```bash
# Activate environment
venv\Scripts\activate

# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_biomechanics.py

# Run with coverage
pytest --cov=app tests/
```

---

## Troubleshooting

### Common Issues

**CUDA out of memory**
- Close other GPU applications
- Reduce batch size in config
- Use smaller model variant
- Lower video resolution

**No person detected**
- Improve lighting in video
- Ensure clear view of player
- Lower confidence threshold
- Check video quality

**LM Studio connection failed**
- Verify LM Studio is running
- Check server is started on port 1234
- Confirm Qwen3.5-9B is loaded
- Check firewall settings

**Slow performance**
- Enable GPU acceleration
- Reduce video resolution
- Use cached results
- Close background applications

**Import errors**
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Check Python version (must be 3.11+)
- Verify CUDA installation

---

## Development

### Setting Up Development Environment

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run linters
black app/
flake8 app/
mypy app/

# Run tests
pytest tests/ -v
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints throughout
- Write docstrings for public methods
- Keep functions focused and testable

### Adding New Features
1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Run full test suite
5. Submit pull request

---

## Roadmap

### v1.0 (Current)
- ✅ Single-camera analysis
- ✅ RTMPose integration
- ✅ Biomechanical features
- ✅ DTW comparison
- ✅ Local LLM coaching
- ✅ PyQt6 interface

### v1.1 (Planned)
- Ball trajectory estimation
- Racket position tracking
- Spin rate approximation
- Advanced statistics dashboard
- Export to video with overlays

### v1.2 (Future)
- Multi-camera support
- 3D reconstruction
- Real-time feedback mode
- Mobile companion app
- Cloud sync option (opt-in)

---

## License

[Specify your license here]

---

## Support

For issues, questions, or feature requests:
- Check documentation in `docs/` folder
- Review troubleshooting section above
- Contact development team

---

**© 2024 TTana - AI Table Tennis Coach**

*Built with ❤️ for table tennis players worldwide*
