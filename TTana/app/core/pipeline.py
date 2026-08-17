"""
Analysis Pipeline Orchestrator
Coordinates the complete video analysis workflow
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import json
import hashlib

from app.core.config import ConfigModel
from app.vision.detector import PersonDetector
from app.vision.pose import PoseEstimator, PoseResult
from app.vision.tracker import ByteTrack, TrackedPerson
from app.vision.smoothing import TemporalSmoother


@dataclass
class AnalysisProgress:
    """Tracks analysis progress."""
    current_step: str = ""
    total_steps: int = 10
    current_step_index: int = 0
    percentage: float = 0.0
    message: str = ""


@dataclass
class VideoAnalysisResult:
    """Complete analysis result for a video."""
    video_path: str
    video_hash: str
    duration: float
    fps: float
    frame_count: int
    
    # Pose data
    keypoints_seq: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_seq: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Tracking data
    track_ids: List[int] = field(default_factory=list)
    bboxes: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding arrays)."""
        return {
            'video_path': self.video_path,
            'video_hash': self.video_hash,
            'duration': self.duration,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'metadata': self.metadata
        }


class AnalysisPipeline:
    """
    Main analysis pipeline orchestrator.
    
    Coordinates all stages of video analysis:
    1. Video loading and validation
    2. Frame extraction
    3. Person detection
    4. Pose estimation
    5. Tracking
    6. Smoothing
    7. Feature extraction
    """

    def __init__(self, config: ConfigModel):
        """
        Initialize pipeline.

        Args:
            config: Application configuration
        """
        self.config = config
        
        # Initialize components
        self.detector = PersonDetector(
            model_path=config.detection.model_path,
            confidence_threshold=config.detection.confidence_threshold,
            nms_threshold=config.detection.nms_threshold,
            device=config.gpu.device if config.gpu.use_cuda else "cpu"
        )
        
        self.pose_estimator = PoseEstimator(
            model_path=config.pose.model_path,
            confidence_threshold=config.pose.confidence_threshold,
            kp_threshold=config.pose.kp_threshold,
            device=config.gpu.device if config.gpu.use_cuda else "cpu"
        )
        
        self.tracker = ByteTrack(
            track_buffer=config.tracking.track_buffer,
            match_threshold=config.tracking.match_threshold,
            confidence_threshold=config.tracking.confidence_threshold
        )
        
        self.smoother = TemporalSmoother(
            method=config.smoothing.method,
            window_length=config.smoothing.window_length,
            polyorder=config.smoothing.polyorder
        )
        
        # Progress tracking
        self.progress = AnalysisProgress()
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, step: str, step_index: int, message: str = ""):
        """Update progress and notify callback."""
        self.progress.current_step = step
        self.progress.current_step_index = step_index
        self.progress.percentage = (step_index / self.progress.total_steps) * 100
        self.progress.message = message
        
        if self.progress_callback:
            self.progress_callback(self.progress)

    def analyze_video(
        self,
        video_path: str,
        reference_video_path: Optional[str] = None
    ) -> Tuple[VideoAnalysisResult, Optional[VideoAnalysisResult]]:
        """
        Analyze user video and optionally compare with reference.

        Args:
            video_path: Path to user video
            reference_video_path: Optional path to reference video

        Returns:
            Tuple of (user_result, reference_result)
        """
        self.progress.total_steps = 10
        self._update_progress("Initializing", 0, "Loading models...")
        
        # Load models
        self.detector.load_model(self.config.detection.model_path)
        self.pose_estimator.load_model(self.config.pose.model_path)
        
        self._update_progress("Processing User Video", 1, f"Loading {Path(video_path).name}...")
        
        # Analyze user video
        user_result = self._analyze_single_video(video_path)
        
        if reference_video_path:
            self._update_progress("Processing Reference Video", 5, f"Loading {Path(reference_video_path).name}...")
            reference_result = self._analyze_single_video(reference_video_path)
        else:
            reference_result = None
        
        self._update_progress("Complete", 10, "Analysis finished!")
        
        return user_result, reference_result

    def _analyze_single_video(self, video_path: str) -> VideoAnalysisResult:
        """
        Analyze a single video.

        Args:
            video_path: Path to video file

        Returns:
            VideoAnalysisResult
        """
        import cv2
        
        # Compute video hash for caching
        video_hash = self._compute_video_hash(video_path)
        
        # Check cache
        if self.config.cache.enabled:
            cached_result = self._load_from_cache(video_hash)
            if cached_result:
                return cached_result
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Storage for results
        all_keypoints = []
        all_confidences = []
        all_bboxes = []
        all_track_ids = []
        
        frame_idx = 0
        self.tracker.reset()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect persons
            detections = self.detector.detect(frame)
            
            if detections:
                # Select target person
                target_det = self.detector.select_target_person(detections)
                
                if target_det:
                    # Estimate pose
                    pose_result = self.pose_estimator.estimate(frame, target_det['bbox'])
                    
                    if pose_result and pose_result.overall_confidence >= self.config.pose.confidence_threshold:
                        # Update tracker
                        det_with_kp = {
                            **target_det,
                            'keypoints': pose_result.keypoint_array
                        }
                        
                        tracks = self.tracker.update([det_with_kp], frame.shape[:2])
                        
                        # Get target track
                        target_track = self.tracker.get_target_track()
                        
                        if target_track and target_track.keypoints is not None:
                            all_keypoints.append(target_track.keypoints[:, :2])  # x, y only
                            all_confidences.append(target_track.keypoints[:, 2])  # confidence
                            all_bboxes.append(target_track.bbox)
                            all_track_ids.append(target_track.track_id)
            
            frame_idx += 1
            
            # Progress update every 10%
            if frame_idx % max(1, frame_count // 10) == 0:
                progress_pct = min(90, 20 + (frame_idx / frame_count) * 70)
                self._update_progress(
                    "Analyzing Frames",
                    int(progress_pct / 10) + 2,
                    f"Frame {frame_idx}/{frame_count}"
                )
        
        cap.release()
        
        # Convert to arrays
        if all_keypoints:
            keypoints_seq = np.array(all_keypoints)
            confidence_seq = np.array(all_confidences)
            bboxes = np.array(all_bboxes)
            
            # Apply smoothing
            smoothed_keypoints = self.smoother.smooth_sequence(
                keypoints_seq,
                confidence_seq
            )
        else:
            smoothed_keypoints = np.array([])
            confidence_seq = np.array([])
            bboxes = np.array([])
        
        # Create result
        result = VideoAnalysisResult(
            video_path=video_path,
            video_hash=video_hash,
            duration=duration,
            fps=fps,
            frame_count=frame_count,
            keypoints_seq=smoothed_keypoints,
            confidence_seq=confidence_seq,
            bboxes=bboxes,
            track_ids=all_track_ids,
            metadata={
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'codec': cap.get(cv2.CAP_PROP_FOURCC)
            }
        )
        
        # Save to cache
        if self.config.cache.enabled:
            self._save_to_cache(result)
        
        return result

    def _compute_video_hash(self, video_path: str) -> str:
        """Compute SHA256 hash of video file."""
        hash_algo = hashlib.sha256()
        
        with open(video_path, 'rb') as f:
            # Read first 1MB for quick hash (or full file for accuracy)
            while chunk := f.read(8192):
                hash_algo.update(chunk)
        
        return hash_algo.hexdigest()

    def _load_from_cache(self, video_hash: str) -> Optional[VideoAnalysisResult]:
        """Load analysis result from cache."""
        cache_dir = Path(self.config.cache.directory)
        cache_path = cache_dir / video_hash / "analysis.json"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct result (simplified - would need to load npz files too)
                return None  # For now, skip cache loading
                
            except Exception:
                pass
        
        return None

    def _save_to_cache(self, result: VideoAnalysisResult):
        """Save analysis result to cache."""
        cache_dir = Path(self.config.cache.directory)
        video_cache_dir = cache_dir / result.video_hash
        video_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = video_cache_dir / "analysis.json"
        with open(metadata_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save arrays as npz
        npz_path = video_cache_dir / "keypoints.npz"
        np.savez(
            npz_path,
            keypoints=result.keypoints_seq,
            confidence=result.confidence_seq,
            bboxes=result.bboxes
        )
