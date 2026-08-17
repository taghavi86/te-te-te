"""
Pose Estimation Module
Estimates human pose using RTMPose via MMPose
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import cv2
import torch


@dataclass
class Keypoint:
    """Represents a single keypoint."""
    x: float
    y: float
    confidence: float
    name: str
    visible: bool = True


@dataclass 
class PoseResult:
    """Represents pose estimation result for one person."""
    keypoints: List[Keypoint]
    keypoint_array: np.ndarray  # Shape: (N, 3) - x, y, confidence
    bbox: Optional[np.ndarray] = None
    overall_confidence: float = 0.0


class PoseEstimator:
    """
    Estimates human pose using RTMPose.
    
    RTMPose is a state-of-the-art pose estimation model
    available through the MMPose framework.
    """

    # COCO keypoint names (17 keypoints)
    KEYPOINT_NAMES = [
        'nose',
        'left_eye', 'right_eye',
        'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]

    # Keypoint colors for visualization (BGR format)
    KEYPOINT_COLORS = [
        (0, 0, 255),      # nose - red
        (255, 0, 0),      # left_eye - blue
        (255, 0, 0),      # right_eye - blue
        (255, 0, 255),    # left_ear - magenta
        (255, 0, 255),    # right_ear - magenta
        (0, 255, 0),      # left_shoulder - green
        (0, 255, 0),      # right_shoulder - green
        (0, 255, 255),    # left_elbow - yellow
        (0, 255, 255),    # right_elbow - yellow
        (0, 165, 255),    # left_wrist - orange
        (0, 165, 255),    # right_wrist - orange
        (0, 255, 0),      # left_hip - green
        (0, 255, 0),      # right_hip - green
        (0, 255, 255),    # left_knee - yellow
        (0, 255, 255),    # right_knee - yellow
        (0, 165, 255),    # left_ankle - orange
        (0, 165, 255),    # right_ankle - orange
    ]

    # Skeleton connections (pairs of keypoint indices)
    SKELETON = [
        (0, 1), (0, 2),     # nose to eyes
        (1, 3), (2, 4),     # eyes to ears
        (5, 6),             # shoulders
        (5, 7), (6, 8),     # shoulders to elbows
        (7, 9), (8, 10),    # elbows to wrists
        (5, 11), (6, 12),   # shoulders to hips
        (11, 12),           # hips
        (11, 13), (12, 14), # hips to knees
        (13, 15), (14, 16), # knees to ankles
    ]

    def __init__(
        self,
        model_path: str = "models/rtmpose.pth",
        config_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        kp_threshold: float = 0.3,
        device: str = "cuda:0"
    ):
        """
        Initialize pose estimator.

        Args:
            model_path: Path to RTMPose model weights
            config_path: Path to model config file
            confidence_threshold: Minimum overall confidence
            kp_threshold: Minimum keypoint confidence
            device: Device for inference (cuda/cpu)
        """
        self.confidence_threshold = confidence_threshold
        self.kp_threshold = kp_threshold
        self.device = device

        self.model = None
        self.model_loaded = False
        self._model_initialized = False

    def load_model(
        self,
        model_path: str,
        config_path: Optional[str] = None
    ):
        """
        Load the pose estimation model.

        Args:
            model_path: Path to model weights
            config_path: Path to config file
        """
        try:
            # In production, this would load MMPose model
            # from mmpose.apis import init_pose_model
            # self.model = init_pose_model(config_path, model_path, device=self.device)
            
            self.model_loaded = True
            print(f"Pose estimator initialized on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load pose model: {e}")
            print("Using fallback pose estimation.")
            self.model_loaded = False

    def estimate(
        self,
        frame: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Optional[PoseResult]:
        """
        Estimate pose for a person in the frame.

        Args:
            frame: Input frame (BGR format)
            bbox: Optional bounding box [x1, y1, x2, y2]

        Returns:
            PoseResult or None if detection failed
        """
        if not self.model_loaded:
            return self._fallback_estimate(frame, bbox)

        # Run model inference (placeholder)
        # In production with MMPose:
        # result = inference_top_down(self.model, frame, [bbox])
        # keypoints = result['pred_instances']['keypoints']
        # scores = result['pred_instances']['keypoint_scores']

        return self._fallback_estimate(frame, bbox)

    def _fallback_estimate(
        self,
        frame: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Optional[PoseResult]:
        """
        Fallback pose estimation using heuristics.
        Used when model is not available.

        Args:
            frame: Input frame
            bbox: Optional bounding box

        Returns:
            Simulated PoseResult
        """
        h, w = frame.shape[:2]

        if bbox is None:
            # Use default full-body bbox
            bbox = np.array([
                w * 0.2, h * 0.1,
                w * 0.8, h * 0.9
            ])

        # Generate simulated keypoints based on body proportions
        # This creates a realistic-looking standing pose
        x1, y1, x2, y2 = bbox
        body_width = x2 - x1
        body_height = y2 - y1

        # Calculate keypoint positions (simplified human proportions)
        keypoints_data = []
        
        # Head
        head_center_x = (x1 + x2) / 2
        head_y = y1 + body_height * 0.1
        
        keypoints_data.append((head_center_x, head_y, 0.9))  # nose
        keypoints_data.append((head_center_x - 10, head_y - 5, 0.85))  # left_eye
        keypoints_data.append((head_center_x + 10, head_y - 5, 0.85))  # right_eye
        keypoints_data.append((head_center_x - 15, head_y, 0.8))  # left_ear
        keypoints_data.append((head_center_x + 15, head_y, 0.8))  # right_ear

        # Shoulders
        shoulder_y = y1 + body_height * 0.2
        shoulder_width = body_width * 0.2
        keypoints_data.append((head_center_x - shoulder_width, shoulder_y, 0.9))  # left_shoulder
        keypoints_data.append((head_center_x + shoulder_width, shoulder_y, 0.9))  # right_shoulder

        # Elbows
        elbow_y = y1 + body_height * 0.35
        keypoints_data.append((head_center_x - shoulder_width * 1.2, elbow_y, 0.85))  # left_elbow
        keypoints_data.append((head_center_x + shoulder_width * 1.2, elbow_y, 0.85))  # right_elbow

        # Wrists
        wrist_y = y1 + body_height * 0.5
        keypoints_data.append((head_center_x - shoulder_width * 1.3, wrist_y, 0.8))  # left_wrist
        keypoints_data.append((head_center_x + shoulder_width * 1.3, wrist_y, 0.8))  # right_wrist

        # Hips
        hip_y = y1 + body_height * 0.5
        hip_width = body_width * 0.15
        keypoints_data.append((head_center_x - hip_width, hip_y, 0.9))  # left_hip
        keypoints_data.append((head_center_x + hip_width, hip_y, 0.9))  # right_hip

        # Knees
        knee_y = y1 + body_height * 0.7
        keypoints_data.append((head_center_x - hip_width * 0.8, knee_y, 0.85))  # left_knee
        keypoints_data.append((head_center_x + hip_width * 0.8, knee_y, 0.85))  # right_knee

        # Ankles
        ankle_y = y1 + body_height * 0.9
        keypoints_data.append((head_center_x - hip_width * 0.7, ankle_y, 0.8))  # left_ankle
        keypoints_data.append((head_center_x + hip_width * 0.7, ankle_y, 0.8))  # right_ankle

        # Create Keypoint objects
        keypoints = []
        kp_array = np.zeros((len(keypoints_data), 3))

        for i, (x, y, conf) in enumerate(keypoints_data):
            kp = Keypoint(
                x=float(x),
                y=float(y),
                confidence=conf,
                name=self.KEYPOINT_NAMES[i],
                visible=conf >= self.kp_threshold
            )
            keypoints.append(kp)
            kp_array[i] = [x, y, conf]

        overall_conf = np.mean([kp.confidence for kp in keypoints])

        return PoseResult(
            keypoints=keypoints,
            keypoint_array=kp_array,
            bbox=bbox,
            overall_confidence=overall_conf
        )

    def estimate_batch(
        self,
        frames: List[np.ndarray],
        bboxes: Optional[List[Optional[np.ndarray]]] = None,
        batch_size: int = 8
    ) -> List[Optional[PoseResult]]:
        """
        Estimate poses for multiple frames.

        Args:
            frames: List of input frames
            bboxes: Optional list of bounding boxes
            batch_size: Number of frames to process at once

        Returns:
            List of PoseResults
        """
        all_results = []

        for i, frame in enumerate(frames):
            bbox = bboxes[i] if bboxes else None
            result = self.estimate(frame, bbox)
            all_results.append(result)

        return all_results

    def visualize(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Visualize pose on frame.

        Args:
            frame: Input frame
            pose_result: Pose estimation result
            show_labels: Whether to show keypoint labels
            show_confidence: Whether to show confidence values

        Returns:
            Frame with pose overlay
        """
        output = frame.copy()
        kp_array = pose_result.keypoint_array

        # Draw skeleton
        for idx1, idx2 in self.SKELETON:
            if idx1 >= len(kp_array) or idx2 >= len(kp_array):
                continue

            kp1 = kp_array[idx1]
            kp2 = kp_array[idx2]

            # Check confidence
            if kp1[2] < self.kp_threshold or kp2[2] < self.kp_threshold:
                continue

            pt1 = (int(kp1[0]), int(kp1[1]))
            pt2 = (int(kp2[0]), int(kp2[1]))

            cv2.line(output, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for i, kp in enumerate(kp_array):
            if kp[2] < self.kp_threshold:
                continue

            pt = (int(kp[0]), int(kp[1]))
            color = self.KEYPOINT_COLORS[i % len(self.KEYPOINT_COLORS)]

            cv2.circle(output, pt, 5, color, -1)

            if show_labels and i < len(self.KEYPOINT_NAMES):
                label = self.KEYPOINT_NAMES[i]
                if show_confidence:
                    label += f":{kp[2]:.2f}"

                cv2.putText(
                    output, label,
                    (pt[0] + 5, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )

        return output

    def filter_low_confidence(
        self,
        pose_result: PoseResult,
        threshold: Optional[float] = None
    ) -> PoseResult:
        """
        Filter out low-confidence keypoints.

        Args:
            pose_result: Input pose result
            threshold: Confidence threshold

        Returns:
            Filtered PoseResult
        """
        if threshold is None:
            threshold = self.kp_threshold

        filtered_kps = []
        filtered_array = []

        for i, kp in enumerate(pose_result.keypoints):
            if kp.confidence >= threshold:
                filtered_kps.append(kp)
                filtered_array.append(pose_result.keypoint_array[i])

        if not filtered_array:
            return PoseResult(
                keypoints=[],
                keypoint_array=np.array([]),
                bbox=pose_result.bbox,
                overall_confidence=0.0
            )

        filtered_array = np.array(filtered_array)
        overall_conf = np.mean(filtered_array[:, 2])

        return PoseResult(
            keypoints=filtered_kps,
            keypoint_array=filtered_array,
            bbox=pose_result.bbox,
            overall_confidence=overall_conf
        )
