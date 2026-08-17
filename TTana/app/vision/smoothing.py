"""
Temporal Smoothing Module
Applies Savitzky-Golay filter and other smoothing techniques to keypoints
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


class TemporalSmoother:
    """
    Applies temporal smoothing to keypoint sequences.
    
    Reduces jitter and noise in pose estimation results
    while preserving motion dynamics.
    """

    def __init__(
        self,
        method: str = "savitzky_golay",
        window_length: int = 7,
        polyorder: int = 2,
        min_samples: int = 3
    ):
        """
        Initialize smoother.

        Args:
            method: Smoothing method ("savitzky_golay", "ema", "uniform")
            window_length: Window size for smoothing (must be odd for SG)
            polyorder: Polynomial order for Savitzky-Golay
            min_samples: Minimum samples required for smoothing
        """
        self.method = method
        self.window_length = window_length
        self.polyorder = polyorder
        self.min_samples = min_samples

        # Ensure window_length is odd for Savitzky-Golay
        if method == "savitzky_golay" and window_length % 2 == 0:
            self.window_length += 1

        # EMA alpha
        self.ema_alpha = 2.0 / (window_length + 1)

        # Buffer for online smoothing
        self.buffer: List[np.ndarray] = []
        self.buffer_size = window_length * 2

    def smooth_sequence(
        self,
        keypoints_seq: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Smooth a sequence of keypoints.

        Args:
            keypoints_seq: Array of shape (T, N, 2) or (T, N, 3)
                          T = time frames, N = number of keypoints
            confidence: Optional confidence weights of shape (T, N)

        Returns:
            Smoothed keypoints array
        """
        if len(keypoints_seq) < self.min_samples:
            return keypoints_seq.copy()

        if self.method == "savitzky_golay":
            return self._savgol_smooth(keypoints_seq, confidence)
        elif self.method == "ema":
            return self._ema_smooth(keypoints_seq, confidence)
        elif self.method == "uniform":
            return self._uniform_smooth(keypoints_seq, confidence)
        else:
            return keypoints_seq.copy()

    def _savgol_smooth(
        self,
        keypoints_seq: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter."""
        seq_len = len(keypoints_seq)
        
        # Adjust window length if sequence is too short
        window = min(self.window_length, seq_len)
        if window % 2 == 0:
            window -= 1
        
        if window < self.polyorder + 2:
            # Can't apply SG filter, return copy
            return keypoints_seq.copy()

        smoothed = np.zeros_like(keypoints_seq)

        # Apply filter to each dimension separately
        if keypoints_seq.ndim == 3:
            # Shape: (T, N, D) where D is 2 or 3
            for d in range(keypoints_seq.shape[2]):
                for n in range(keypoints_seq.shape[1]):
                    column = keypoints_seq[:, n, d]
                    
                    # Weight by confidence if available
                    if confidence is not None:
                        # Replace low-confidence points with interpolated values
                        col_conf = confidence[:, n]
                        mask = col_conf < 0.5
                        if mask.any() and not mask.all():
                            column = column.copy()
                            valid_idx = np.where(~mask)[0]
                            invalid_idx = np.where(mask)[0]
                            
                            if len(valid_idx) > 1:
                                # Linear interpolation for low-confidence points
                                column[invalid_idx] = np.interp(
                                    invalid_idx,
                                    valid_idx,
                                    column[valid_idx]
                                )
                    
                    smoothed[:, n, d] = savgol_filter(
                        column,
                        window_length=window,
                        polyorder=self.polyorder,
                        mode='nearest'
                    )
        else:
            # Shape: (T, N, 2) - simple case
            for n in range(keypoints_seq.shape[1]):
                for d in range(keypoints_seq.shape[2]):
                    smoothed[:, n, d] = savgol_filter(
                        keypoints_seq[:, n, d],
                        window_length=window,
                        polyorder=self.polyorder,
                        mode='nearest'
                    )

        return smoothed

    def _ema_smooth(
        self,
        keypoints_seq: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply Exponential Moving Average smoothing."""
        smoothed = np.zeros_like(keypoints_seq)
        alpha = self.ema_alpha

        # Initialize with first value
        smoothed[0] = keypoints_seq[0]

        for t in range(1, len(keypoints_seq)):
            if confidence is not None:
                # Adaptive alpha based on confidence
                conf_weight = confidence[t].mean()
                adaptive_alpha = alpha * conf_weight
            else:
                adaptive_alpha = alpha

            smoothed[t] = (
                adaptive_alpha * keypoints_seq[t] +
                (1 - adaptive_alpha) * smoothed[t - 1]
            )

        return smoothed

    def _uniform_smooth(
        self,
        keypoints_seq: np.ndarray,
        confidence: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply uniform (moving average) filter."""
        if keypoints_seq.ndim == 3:
            smoothed = np.zeros_like(keypoints_seq)
            for d in range(keypoints_seq.shape[2]):
                for n in range(keypoints_seq.shape[1]):
                    smoothed[:, n, d] = uniform_filter1d(
                        keypoints_seq[:, n, d],
                        size=self.window_length,
                        mode='nearest'
                    )
            return smoothed
        else:
            return uniform_filter1d(
                keypoints_seq,
                size=self.window_length,
                mode='nearest',
                axis=0
            )

    def smooth_online(self, new_keypoints: np.ndarray) -> np.ndarray:
        """
        Add new keypoints and return smoothed result.
        For real-time applications.

        Args:
            new_keypoints: New keypoint frame of shape (N, 2) or (N, 3)

        Returns:
            Smoothed keypoints
        """
        # Add to buffer
        self.buffer.append(new_keypoints.copy())

        # Keep buffer bounded
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Need minimum samples
        if len(self.buffer) < self.min_samples:
            return new_keypoints.copy()

        # Convert to array
        buffer_array = np.array(self.buffer)

        # Smooth only the most recent window
        window_start = max(0, len(self.buffer) - self.window_length)
        window_data = buffer_array[window_start:]

        smoothed_window = self.smooth_sequence(window_data)

        # Return the last smoothed frame
        return smoothed_window[-1]

    def reset(self):
        """Reset internal buffer."""
        self.buffer = []

    def compute_velocity(
        self,
        keypoints_seq: np.ndarray,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Compute velocity from smoothed keypoints.

        Args:
            keypoints_seq: Keypoint sequence (T, N, 2)
            fps: Frames per second

        Returns:
            Velocity array (T, N, 2)
        """
        # First smooth the input
        smoothed = self.smooth_sequence(keypoints_seq)

        # Compute finite difference
        velocity = np.zeros_like(smoothed)
        dt = 1.0 / fps

        velocity[1:-1] = (smoothed[2:] - smoothed[:-2]) / (2 * dt)
        
        # Handle boundaries
        velocity[0] = (smoothed[1] - smoothed[0]) / dt
        velocity[-1] = (smoothed[-1] - smoothed[-2]) / dt

        return velocity

    def compute_acceleration(
        self,
        keypoints_seq: np.ndarray,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Compute acceleration from smoothed keypoints.

        Args:
            keypoints_seq: Keypoint sequence (T, N, 2)
            fps: Frames per second

        Returns:
            Acceleration array (T, N, 2)
        """
        velocity = self.compute_velocity(keypoints_seq, fps)
        return self.compute_velocity_from_frames(velocity, fps)

    def compute_velocity_from_frames(
        self,
        positions: np.ndarray,
        fps: float = 30.0
    ) -> np.ndarray:
        """Compute velocity from position frames."""
        velocity = np.zeros_like(positions)
        dt = 1.0 / fps

        velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        velocity[0] = (positions[1] - positions[0]) / dt
        velocity[-1] = (positions[-1] - positions[-2]) / dt

        return velocity


def interpolate_keypoints(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Interpolate low-confidence keypoints.

    Args:
        keypoints: Keypoint array (N, 2) or (T, N, 2)
        confidence: Confidence array (N,) or (T, N)
        threshold: Confidence threshold

    Returns:
        Interpolated keypoints
    """
    if keypoints.ndim == 2:
        # Single frame
        return _interpolate_1d(keypoints, confidence, threshold)
    else:
        # Sequence
        result = np.zeros_like(keypoints)
        for t in range(len(keypoints)):
            result[t] = _interpolate_1d(keypoints[t], confidence[t], threshold)
        return result


def _interpolate_1d(
    keypoints: np.ndarray,
    confidence: np.ndarray,
    threshold: float
) -> np.ndarray:
    """Interpolate keypoints in a single frame."""
    result = keypoints.copy()
    mask = confidence < threshold

    if not mask.any() or mask.all():
        return result

    valid_idx = np.where(~mask)[0]
    invalid_idx = np.where(mask)[0]

    if len(valid_idx) < 2:
        # Not enough valid points for interpolation
        return result

    # Interpolate each dimension
    for d in range(keypoints.shape[1]):
        result[invalid_idx, d] = np.interp(
            invalid_idx,
            valid_idx,
            keypoints[valid_idx, d]
        )

    return result
