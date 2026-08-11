"""
High-Precision Ball Detection and Tracking
Uses YOLOv8 with custom training for tennis ball detection
Implements Kalman Filter for smooth trajectory prediction
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
from collections import deque
import torch

class BallDetector:
    """Advanced tennis ball detection and tracking system"""
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Initialize YOLOv8 for ball detection
        # Using the largest model for maximum accuracy
        model_path = config.get('YOLO_MODEL', 'yolov8x.pt')
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Tennis ball class index (may need custom training)
        self.ball_class_id = 32  # Sports ball in COCO dataset
        
        # Tracking parameters
        self.tracking_history = config.get('BALL_TRACKING_HISTORY', 30)
        self.confidence_threshold = config.get('CONFIDENCE_THRESHOLD', 0.75)
        
        # Kalman Filter for smooth tracking
        self.kalman_filter = self._init_kalman_filter()
        self.ball_trajectory = deque(maxlen=self.tracking_history)
        
        # Motion prediction
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
    
    def _init_kalman_filter(self):
        """Initialize Kalman Filter for ball tracking"""
        # State: [x, y, vx, vy]
        kalman = cv2.KalmanFilter(4, 2)
        
        # Transition matrix
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        return kalman
    
    def detect(self, frame: np.ndarray) -> Dict:
        """Detect tennis ball in frame"""
        
        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        
        ball_detected = False
        ball_position = None
        ball_confidence = 0.0
        ball_bbox = None
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # Check if this is a sports ball
                cls_id = int(box.cls[0])
                if cls_id == self.ball_class_id:
                    confidence = float(box.conf[0])
                    
                    if confidence > self.confidence_threshold:
                        ball_detected = True
                        ball_confidence = confidence
                        
                        # Extract bounding box
                        bbox = box.xyxy[0].cpu().numpy()
                        ball_bbox = bbox
                        
                        # Calculate center position
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_center = (bbox[1] + bbox[3]) / 2
                        ball_position = np.array([x_center, y_center])
                        
                        break
        
        # Update Kalman Filter if ball detected
        if ball_detected and ball_position is not None:
            # Predict
            predicted_state = self.kalman_filter.predict()
            
            # Update with measurement
            measurement = ball_position.reshape(-1, 1).astype(np.float32)
            self.kalman_filter.correct(measurement)
            
            # Store trajectory
            self.ball_trajectory.append(ball_position)
            
            # Calculate velocity and acceleration
            if len(self.ball_trajectory) >= 2:
                prev_pos = self.ball_trajectory[-2]
                self.velocity = ball_position - prev_pos
                
                if len(self.ball_trajectory) >= 3:
                    prev_vel = self.velocity
                    self.velocity = ball_position - prev_pos
                    self.acceleration = self.velocity - prev_vel
        
        # Get smoothed position from Kalman Filter
        if not ball_detected and len(self.ball_trajectory) > 0:
            # Use last known position with prediction
            kalman_state = self.kalman_filter.statePost
            predicted_pos = kalman_state[:2].flatten()
            
            return {
                'detected': False,
                'position': predicted_pos,
                'confidence': 0.0,
                'bbox': None,
                'predicted': True,
                'velocity': self.velocity,
                'acceleration': self.acceleration
            }
        
        return {
            'detected': ball_detected,
            'position': ball_position,
            'confidence': ball_confidence,
            'bbox': ball_bbox,
            'predicted': False,
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'trajectory': list(self.ball_trajectory)
        }
    
    def predict_ball_landing(self, frame_height: int) -> Optional[np.ndarray]:
        """Predict where the ball will land based on current trajectory"""
        
        if len(self.ball_trajectory) < 3:
            return None
        
        positions = np.array(self.ball_trajectory)
        
        # Fit parabola to trajectory
        if len(positions) >= 3:
            # Use last few points for prediction
            recent_points = positions[-min(10, len(positions)):]
            
            # Fit quadratic curve to y-coordinates
            x_coords = recent_points[:, 0]
            y_coords = recent_points[:, 1]
            
            try:
                # Polynomial fit: y = ax^2 + bx + c
                coeffs = np.polyfit(x_coords, y_coords, 2)
                
                # Find landing point (where y = frame_height)
                a, b, c = coeffs
                discriminant = b**2 - 4*a*(c - frame_height)
                
                if discriminant >= 0:
                    x_land1 = (-b + np.sqrt(discriminant)) / (2*a)
                    x_land2 = (-b - np.sqrt(discriminant)) / (2*a)
                    
                    # Choose the point in direction of motion
                    if self.velocity[0] > 0:
                        x_landing = max(x_land1, x_land2)
                    else:
                        x_landing = min(x_land1, x_land2)
                    
                    return np.array([x_landing, frame_height])
            except:
                pass
        
        return None
    
    def calculate_spin_estimate(self) -> Dict:
        """Estimate ball spin from trajectory curvature"""
        
        if len(self.ball_trajectory) < 5:
            return {'topspin': 0, 'backspin': 0, 'sidespin': 0}
        
        positions = np.array(self.ball_trajectory)
        
        # Analyze trajectory deviation from parabola
        x_coords = positions[:, 0]
        y_coords = positions[:, 1]
        
        # Fit expected parabola (gravity only)
        expected_coeffs = np.polyfit(x_coords, y_coords, 2)
        expected_y = np.polyval(expected_coeffs, x_coords)
        
        # Calculate deviation
        deviation = y_coords - expected_y
        
        # Topspin: ball drops faster than expected
        # Backspin: ball stays in air longer
        avg_deviation = np.mean(deviation)
        
        # Sidespin: lateral deviation
        if len(x_coords) > 1:
            x_velocity = np.diff(x_coords)
            lateral_acceleration = np.diff(x_velocity)
            sidespin = np.mean(lateral_acceleration) * 100
        else:
            sidespin = 0
        
        return {
            'topspin': max(0, -avg_deviation * 10),
            'backspin': max(0, avg_deviation * 10),
            'sidespin': sidespin
        }
    
    def reset_tracking(self):
        """Reset tracking state for new sequence"""
        self.ball_trajectory.clear()
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.kalman_filter.statePost = np.zeros((4, 1), dtype=np.float32)
