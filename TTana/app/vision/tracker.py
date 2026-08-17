"""
Multi-Object Tracking Module
Tracks persons across frames using ByteTrack with Kalman Filter
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

import cv2


@dataclass
class TrackedPerson:
    """Represents a tracked person."""
    track_id: int
    bbox: np.ndarray
    keypoints: Optional[np.ndarray] = None
    confidence: float = 0.0
    age: int = 0
    time_since_update: int = 0
    history: deque = None  # Keypoint history for smoothing

    def __post_init__(self):
        if self.history is None:
            self.history = deque(maxlen=30)


class ByteTrack:
    """
    Simple ByteTrack implementation for multi-object tracking.
    
    Combines high and low confidence detections to maintain
    robust tracking even during occlusions.
    """

    def __init__(
        self,
        track_buffer: int = 30,
        match_threshold: float = 0.8,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize tracker.

        Args:
            track_buffer: Frames to keep track before removing
            match_threshold: IoU threshold for matching
            confidence_threshold: Minimum detection confidence
        """
        self.track_buffer = track_buffer
        self.match_threshold = match_threshold
        self.confidence_threshold = confidence_threshold

        self.tracks: List[TrackedPerson] = []
        self.next_track_id = 0

        # Kalman filter parameters (simplified)
        self.process_noise = 1e-4
        self.measurement_noise = 1e-1

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int]
    ) -> List[TrackedPerson]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections from current frame
            frame_shape: (height, width) of frame

        Returns:
            List of updated tracks
        """
        # Separate high and low confidence detections
        high_conf_dets = []
        low_conf_dets = []

        for det in detections:
            if det['confidence'] >= self.confidence_threshold:
                high_conf_dets.append(det)
            else:
                low_conf_dets.append(det)

        # Get active tracks
        active_tracks = [t for t in self.tracks if t.time_since_update < self.track_buffer]

        # First association with high confidence detections
        if high_conf_dets and active_tracks:
            matched, unmatched_tracks, unmatched_dets = self._associate(
                active_tracks, high_conf_dets, frame_shape
            )

            # Update matched tracks
            for track_idx, det_idx in matched:
                track = active_tracks[track_idx]
                det = high_conf_dets[det_idx]

                self._update_track(track, det)

            # Try to match unmatched tracks with low confidence detections
            if unmatched_tracks and low_conf_dets:
                low_matched, low_unmatched_tracks, _ = self._associate(
                    [active_tracks[i] for i in unmatched_tracks],
                    low_conf_dets,
                    frame_shape,
                    threshold=self.match_threshold * 0.5
                )

                for tm_idx, det_idx in low_matched:
                    track_idx = unmatched_tracks[tm_idx]
                    track = active_tracks[track_idx]
                    det = low_conf_dets[det_idx]

                    self._update_track(track, det)
                    unmatched_tracks.remove(track_idx)

            # Handle unmatched tracks
            for track_idx in unmatched_tracks:
                track = active_tracks[track_idx]
                track.time_since_update += 1

            # Handle unmatched detections (new tracks)
            for det_idx in unmatched_dets:
                det = high_conf_dets[det_idx]
                self._create_new_track(det)

        else:
            # No detections or no tracks - just age existing tracks
            for track in active_tracks:
                track.time_since_update += 1

        # Remove lost tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update < self.track_buffer
        ]

        return self.tracks

    def _associate(
        self,
        tracks: List[TrackedPerson],
        detections: List[Dict[str, Any]],
        frame_shape: Tuple[int, int],
        threshold: Optional[float] = None
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.

        Returns:
            matched: List of (track_idx, det_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_dets: List of unmatched detection indices
        """
        if threshold is None:
            threshold = self.match_threshold

        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Build cost matrix (IoU)
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det['bbox'])
                cost_matrix[i, j] = iou

        # Greedy matching
        matched = []
        used_tracks = set()
        used_dets = set()

        while True:
            if cost_matrix.size == 0:
                break

            max_iou = np.max(cost_matrix)
            if max_iou < threshold:
                break

            track_idx, det_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)

            if track_idx not in used_tracks and det_idx not in used_dets:
                matched.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_dets.add(det_idx)

            # Remove used row and column
            cost_matrix = np.delete(cost_matrix, track_idx, axis=0)
            cost_matrix = np.delete(cost_matrix, 0, axis=1) if det_idx == 0 else np.delete(cost_matrix, det_idx, axis=1)

            # Adjust indices for remaining items
            adjusted_matched = []
            for t_idx, d_idx in matched:
                adj_t = t_idx
                adj_d = d_idx
                for removed_t in sorted([m[0] for m in matched if m != (t_idx, d_idx)]):
                    if removed_t < t_idx:
                        adj_t -= 1
                adjusted_matched.append((adj_t, adj_d))
            matched = adjusted_matched

        # Simpler approach - rebuild from scratch
        matched = []
        used_tracks = set()
        used_dets = set()

        # Sort by IoU descending
        iou_pairs = []
        for i in range(len(tracks)):
            for j in range(len(detections)):
                iou = self._compute_iou(tracks[i].bbox, detections[j]['bbox'])
                if iou >= threshold:
                    iou_pairs.append((iou, i, j))

        iou_pairs.sort(reverse=True)

        for iou, track_idx, det_idx in iou_pairs:
            if track_idx not in used_tracks and det_idx not in used_dets:
                matched.append((track_idx, det_idx))
                used_tracks.add(track_idx)
                used_dets.add(det_idx)

        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in used_dets]

        return matched, unmatched_tracks, unmatched_dets

    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _update_track(self, track: TrackedPerson, detection: Dict[str, Any]):
        """Update track with new detection."""
        # Update bbox (simple replacement, could use Kalman filter)
        track.bbox = detection['bbox'].copy()
        track.confidence = detection['confidence']
        track.time_since_update = 0
        track.age += 1

        # Store keypoints in history if available
        if 'keypoints' in detection and detection['keypoints'] is not None:
            track.keypoints = detection['keypoints'].copy()
            track.history.append(track.keypoints.copy())

    def _create_new_track(self, detection: Dict[str, Any]):
        """Create a new track from detection."""
        track = TrackedPerson(
            track_id=self.next_track_id,
            bbox=detection['bbox'].copy(),
            keypoints=detection.get('keypoints'),
            confidence=detection['confidence'],
            age=1,
            time_since_update=0
        )

        if track.keypoints is not None:
            track.history.append(track.keypoints.copy())

        self.tracks.append(track)
        self.next_track_id += 1

    def get_target_track(
        self,
        strategy: str = "largest"
    ) -> Optional[TrackedPerson]:
        """
        Get target track based on strategy.

        Args:
            strategy: Selection strategy

        Returns:
            Selected track or None
        """
        if not self.tracks:
            return None

        if strategy == "largest":
            return max(
                self.tracks,
                key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
            )
        elif strategy == "highest_confidence":
            return max(self.tracks, key=lambda t: t.confidence)
        elif strategy == "first":
            return self.tracks[0]
        else:
            return self.tracks[0]

    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.next_track_id = 0
