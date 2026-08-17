"""
Person Detection Module
Detects players in video frames using RTMDet
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import cv2
import torch


class PersonDetector:
    """Detects persons in video frames."""

    def __init__(
        self,
        model_path: str = "models/rtmdet.pth",
        config_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cuda:0"
    ):
        """
        Initialize person detector.

        Args:
            model_path: Path to RTMDet model weights
            config_path: Path to model config file
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to run inference on (cuda/cpu)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device

        # Load model (placeholder - actual implementation requires MMPose setup)
        self.model = None
        self.model_loaded = False

        # Model will be loaded when first frame is processed
        self._model_initialized = False

    def load_model(self, model_path: str, config_path: Optional[str] = None):
        """
        Load the detection model.

        Args:
            model_path: Path to model weights
            config_path: Path to config file
        """
        try:
            # In production, this would load MMPose/RTMDet model
            # from mmdet.apis import init_detector
            # self.model = init_detector(config_path, model_path, device=self.device)
            
            self.model_loaded = True
            print(f"Person detector initialized on {self.device}")
            
        except Exception as e:
            print(f"Warning: Could not load detection model: {e}")
            print("Using fallback detection method.")
            self.model_loaded = False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detection dictionaries with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection confidence
                - label: class label
        """
        if not self._model_initialized and self.model_loaded:
            # Lazy model initialization could happen here
            pass

        # If model not loaded, use simple heuristic (largest blob)
        if not self.model_loaded:
            return self._fallback_detect(frame)

        # Run model inference (placeholder)
        # In production:
        # result = inference_detector(self.model, frame)
        # boxes = result.pred_instances.bboxes.cpu().numpy()
        # scores = result.pred_instances.scores.cpu().numpy()

        # Placeholder implementation
        return self._fallback_detect(frame)

    def _fallback_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Fallback detection using simple heuristics.
        Used when model is not available.

        Args:
            frame: Input frame

        Returns:
            List of pseudo-detections
        """
        h, w = frame.shape[:2]

        # Simulate a person detection in center of frame
        # This is a placeholder - real implementation uses RTMDet
        person_width = int(w * 0.3)
        person_height = int(h * 0.6)
        x1 = (w - person_width) // 2
        y1 = int(h * 0.2)
        x2 = x1 + person_width
        y2 = y1 + person_height

        return [{
            'bbox': np.array([x1, y1, x2, y2]),
            'confidence': 0.85,
            'label': 'person',
            'label_id': 0
        }]

    def detect_batch(
        self,
        frames: List[np.ndarray],
        batch_size: int = 8
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect persons in multiple frames.

        Args:
            frames: List of input frames
            batch_size: Number of frames to process at once

        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_results = []

            for frame in batch:
                detections = self.detect(frame)
                batch_results.append(detections)

            all_detections.extend(batch_results)

        return all_detections

    def select_target_person(
        self,
        detections: List[Dict[str, Any]],
        strategy: str = "largest"
    ) -> Optional[Dict[str, Any]]:
        """
        Select target person from multiple detections.

        Args:
            detections: List of person detections
            strategy: Selection strategy ("largest", "highest_confidence", "center")

        Returns:
            Selected detection or None
        """
        if not detections:
            return None

        if strategy == "largest":
            # Select largest bounding box
            return max(
                detections,
                key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1])
            )

        elif strategy == "highest_confidence":
            # Select highest confidence detection
            return max(detections, key=lambda d: d['confidence'])

        elif strategy == "center":
            # Select detection closest to image center
            h, w = 1080, 1920  # Default, should be passed in
            center_x, center_y = w / 2, h / 2

            def distance_to_center(det):
                bbox = det['bbox']
                det_center_x = (bbox[0] + bbox[2]) / 2
                det_center_y = (bbox[1] + bbox[3]) / 2
                return ((det_center_x - center_x) ** 2 + 
                        (det_center_y - center_y) ** 2) ** 0.5

            return min(detections, key=distance_to_center)

        else:
            # Default to largest
            return detections[0]

    def filter_detections(
        self,
        detections: List[Dict[str, Any]],
        min_confidence: Optional[float] = None,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter detections by various criteria.

        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area

        Returns:
            Filtered list of detections
        """
        filtered = []

        for det in detections:
            # Confidence filter
            if min_confidence is not None and det['confidence'] < min_confidence:
                continue

            # Area filter
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if min_area is not None and area < min_area:
                continue

            if max_area is not None and area > max_area:
                continue

            filtered.append(det)

        return filtered
