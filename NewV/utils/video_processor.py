"""
Utility modules for Tennis AI Coach
Logging, video processing, and data collection utilities
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

# Logger setup
def setup_logger(name: str) -> logging.Logger:
    """Setup configured logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get existing logger or create new one"""
    return logging.getLogger(name)


class VideoProcessor:
    """Advanced video processing utilities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.resolution = config.get('RESOLUTION', (1920, 1080))
        self.fps = config.get('FRAME_RATE', 60)
    
    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """Load video with optimized settings"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        return cap
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for AI models"""
        
        # Resize to optimal resolution
        frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LINEAR)
        
        # Enhance contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        lab[:, :, 0] = enhanced_l
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Denoise
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        return frame
    
    def extract_frames(self, cap: cv2.VideoCapture, 
                      step: int = 1) -> List[np.ndarray]:
        """Extract frames from video"""
        frames = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % step == 0:
                frames.append(self.preprocess_frame(frame))
            
            frame_count += 1
        
        return frames
    
    def save_processed_video(self, frames: List[np.ndarray], 
                            output_path: str):
        """Save processed frames as video"""
        
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
    
    def overlay_analysis(self, frame: np.ndarray, 
                        analysis_data: Dict) -> np.ndarray:
        """Overlay analysis data on frame"""
        
        overlay = frame.copy()
        
        # Draw keypoints
        if 'pose' in analysis_data and analysis_data['pose'].get('detected'):
            keypoints = analysis_data['pose'].get('keypoints', [])
            
            for i, kp in enumerate(keypoints):
                if kp.get('visibility', 0) > 0.5:
                    x = int(kp['x'] * frame.shape[1])
                    y = int(kp['y'] * frame.shape[0])
                    
                    cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        
        # Draw ball position
        if 'ball' in analysis_data and analysis_data['ball'].get('detected'):
            position = analysis_data['ball'].get('position')
            if position is not None:
                x, y = int(position[0]), int(position[1])
                cv2.circle(overlay, (x, y), 10, (255, 0, 0), -1)
        
        # Draw technique score
        if 'technique' in analysis_data:
            score = analysis_data['technique'].get('score', 0)
            
            cv2.putText(
                overlay,
                f"Score: {score:.1f}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
        
        return overlay


class DataCollector:
    """Collect and store analysis data"""
    
    def __init__(self, output_dir: str = "data/analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_buffer = []
    
    def add_frame_data(self, frame_number: int, data: Dict):
        """Add frame-level data"""
        self.data_buffer.append({
            'frame': frame_number,
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
    
    def save_session(self, filename: Optional[str] = None):
        """Save session data to file"""
        
        if not filename:
            filename = f"session_{self.session_id}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'total_frames': len(self.data_buffer),
                'data': self.data_buffer
            }, f, indent=2)
        
        return output_path
    
    def export_statistics(self, results: Dict) -> Path:
        """Export analysis statistics"""
        
        stats_path = self.output_dir / f"stats_{self.session_id}.json"
        
        with open(stats_path, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'statistics': results.get('statistics', {}),
                'metadata': results.get('metadata', {})
            }, f, indent=2)
        
        return stats_path
    
    def clear_buffer(self):
        """Clear data buffer"""
        self.data_buffer = []


def calculate_angle(p1: np.ndarray, p2: np.ndarray, 
                   p3: np.ndarray) -> float:
    """Calculate angle between three points"""
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Normalize
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
    
    # Dot product
    dot = np.dot(v1_norm, v2_norm)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    return np.degrees(angle)


def smooth_trajectory(trajectory: List[np.ndarray], 
                     window_size: int = 5) -> List[np.ndarray]:
    """Smooth trajectory using moving average"""
    
    if len(trajectory) < window_size:
        return trajectory
    
    smoothed = []
    trajectory_array = np.array(trajectory)
    
    for i in range(len(trajectory)):
        start = max(0, i - window_size // 2)
        end = min(len(trajectory), i + window_size // 2 + 1)
        
        avg = np.mean(trajectory_array[start:end], axis=0)
        smoothed.append(avg)
    
    return smoothed


def detect_scene_changes(frames: List[np.ndarray], 
                        threshold: float = 0.3) -> List[int]:
    """Detect scene changes in video"""
    
    if len(frames) < 2:
        return []
    
    changes = []
    
    for i in range(1, len(frames)):
        # Calculate frame difference
        diff = cv2.absdiff(frames[i-1], frames[i])
        diff_ratio = np.sum(diff) / (frames[i].size * 255)
        
        if diff_ratio > threshold:
            changes.append(i)
    
    return changes
