"""
Core Pipeline for Tennis Video Analysis
Orchestrates all AI models and analysis components
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tempfile
import time

from models.pose_estimator import PoseEstimator
from models.ball_detector import BallDetector
from models.racket_detector import RacketDetector
from models.technique_analyzer import TechniqueAnalyzer
from models.pro_comparator import ProfessionalComparator
from utils.video_processor import VideoProcessor
from utils.data_collector import DataCollector
from utils.logger import get_logger

logger = get_logger(__name__)

class TennisAnalysisPipeline:
    """Main pipeline coordinating all analysis components"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() and config.get('GPU_ACCELERATION', True) else 'cpu'
        
        # Initialize models
        logger.info(f"Initializing models on {self.device}")
        self.pose_estimator = PoseEstimator(config, self.device)
        self.ball_detector = BallDetector(config, self.device)
        self.racket_detector = RacketDetector(config, self.device)
        self.technique_analyzer = TechniqueAnalyzer(config)
        self.pro_comparator = ProfessionalComparator(config)
        
        # Utilities
        self.video_processor = VideoProcessor(config)
        self.data_collector = DataCollector()
        
        # State
        self.current_frame = 0
        self.total_frames = 0
        self.analysis_results = {}
    
    def analyze_video(self, video_file, analysis_options: List[str], 
                     pro_video=None) -> Dict:
        """Complete video analysis pipeline"""
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name
        
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Processing video: {self.total_frames} frames at {fps} FPS")
            
            # Storage for results
            frame_results = []
            trajectory_data = {
                'ball': [],
                'racket': [],
                'joints': []
            }
            
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                self.current_frame = frame_count
                
                # Progress logging
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"Processed {frame_count}/{self.total_frames} frames ({elapsed:.1f}s)")
                
                # Run analysis based on selected options
                frame_data = {'frame': frame_count, 'timestamp': frame_count/fps}
                
                if 'Pose Detection' in analysis_options:
                    pose_data = self.pose_estimator.detect(frame)
                    frame_data['pose'] = pose_data
                    trajectory_data['joints'].append(pose_data.get('keypoints', []))
                
                if 'Ball Tracking' in analysis_options:
                    ball_data = self.ball_detector.detect(frame)
                    frame_data['ball'] = ball_data
                    if ball_data.get('detected'):
                        trajectory_data['ball'].append(ball_data['position'])
                
                if 'Racket Detection' in analysis_options:
                    racket_data = self.racket_detector.detect(frame)
                    frame_data['racket'] = racket_data
                    if racket_data.get('detected'):
                        trajectory_data['racket'].append(racket_data['position'])
                
                # Advanced analysis
                if 'Technique Analysis' in analysis_options and 'pose' in frame_data:
                    technique_data = self.technique_analyzer.analyze(
                        frame_data['pose'], 
                        frame_count
                    )
                    frame_data['technique'] = technique_data
                
                frame_results.append(frame_data)
                
                # Release memory periodically
                if frame_count % 100 == 0:
                    torch.cuda.empty_cache() if self.device == 'cuda' else None
            
            cap.release()
            
            # Post-processing
            logger.info("Running post-processing analysis...")
            
            # Trajectory smoothing
            trajectory_data = self._smooth_trajectories(trajectory_data)
            
            # Professional comparison if provided
            if pro_video and 'Professional Comparison' in analysis_options:
                comparison = self.pro_comparator.compare(
                    frame_results, 
                    pro_video,
                    trajectory_data
                )
                self.analysis_results['comparison'] = comparison
            
            # Aggregate statistics
            statistics = self._calculate_statistics(frame_results, trajectory_data)
            
            self.analysis_results = {
                'frames': frame_results,
                'trajectories': trajectory_data,
                'statistics': statistics,
                'metadata': {
                    'total_frames': self.total_frames,
                    'fps': fps,
                    'duration': self.total_frames / fps,
                    'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                }
            }
            
            logger.info("Analysis complete!")
            return self.analysis_results
            
        finally:
            # Cleanup
            Path(video_path).unlink(missing_ok=True)
    
    def _smooth_trajectories(self, trajectory_data: Dict) -> Dict:
        """Apply smoothing filters to trajectories"""
        from scipy.signal import savgol_filter
        
        smoothed = {}
        window_size = min(11, len(trajectory_data.get('ball', [])))
        if window_size >= 5:
            ball_traj = np.array(trajectory_data.get('ball', []))
            if len(ball_traj) > 0:
                smoothed['ball'] = savgol_filter(ball_traj, window_size, 3, axis=0)
        
        return {**trajectory_data, **smoothed}
    
    def _calculate_statistics(self, frame_results: List, 
                            trajectory_data: Dict) -> Dict:
        """Calculate aggregate statistics from analysis"""
        
        stats = {
            'ball': {},
            'pose': {},
            'technique': {}
        }
        
        # Ball statistics
        if trajectory_data.get('ball'):
            ball_positions = np.array(trajectory_data['ball'])
            stats['ball'] = {
                'max_speed': self._calculate_speed(ball_positions),
                'trajectory_length': len(ball_positions),
                'coverage_area': self._calculate_coverage(ball_positions)
            }
        
        # Pose statistics
        joint_angles = []
        for frame in frame_results:
            if 'pose' in frame and 'angles' in frame['pose']:
                joint_angles.append(frame['pose']['angles'])
        
        if joint_angles:
            stats['pose'] = {
                'avg_angles': np.mean(joint_angles, axis=0).tolist(),
                'angle_variance': np.var(joint_angles, axis=0).tolist()
            }
        
        # Technique statistics
        technique_scores = []
        for frame in frame_results:
            if 'technique' in frame:
                technique_scores.append(frame['technique'].get('score', 0))
        
        if technique_scores:
            stats['technique'] = {
                'average_score': np.mean(technique_scores),
                'consistency': 1 - np.std(technique_scores) / (np.mean(technique_scores) + 1e-6),
                'peak_score': max(technique_scores)
            }
        
        return stats
    
    def _calculate_speed(self, positions: np.ndarray) -> float:
        """Calculate maximum speed from position sequence"""
        if len(positions) < 2:
            return 0.0
        
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        return float(np.max(speeds))
    
    def _calculate_coverage(self, positions: np.ndarray) -> float:
        """Calculate area covered by ball trajectory"""
        if len(positions) < 3:
            return 0.0
        
        # Convex hull area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(positions[:, :2])  # Use x,y coordinates
            return float(hull.volume)  # In 2D, volume is area
        except:
            return 0.0
