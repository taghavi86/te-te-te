"""
Racket Detection and Swing Analysis
Uses YOLOv8 with custom training for tennis racket detection
Analyzes swing mechanics and racket positioning
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from ultralytics import YOLO
from collections import deque
import torch

class RacketDetector:
    """Advanced tennis racket detection and swing analysis"""
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Initialize YOLOv8 for racket detection
        model_path = config.get('YOLO_MODEL', 'yolov8x.pt')
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Racket class index (custom trained model would have specific class)
        # For now, we'll use a combination of approaches
        self.racket_class_id = None  # Custom class ID for tennis racket
        
        # Tracking parameters
        self.confidence_threshold = config.get('CONFIDENCE_THRESHOLD', 0.75)
        self.tracking_history = 20
        
        # Racket state tracking
        self.racket_trajectory = deque(maxlen=self.tracking_history)
        self.swing_phases = []
        
        # Swing analysis
        self.swing_speed = 0.0
        self.swing_acceleration = 0.0
        self.racket_angle = 0.0
    
    def detect(self, frame: np.ndarray) -> Dict:
        """Detect tennis racket in frame"""
        
        # Run YOLO inference
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        
        racket_detected = False
        racket_position = None
        racket_confidence = 0.0
        racket_bbox = None
        racket_orientation = 0.0
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                # Look for objects that could be rackets
                # In a custom model, this would be a specific class
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # For demonstration, we'll consider any detected object
                # In practice, you'd train a custom model for rackets
                if confidence > self.confidence_threshold:
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Calculate aspect ratio to identify racket-like shapes
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    aspect_ratio = height / (width + 1e-6)
                    
                    # Tennis rackets are typically elongated
                    if 1.5 < aspect_ratio < 4.0:
                        racket_detected = True
                        racket_confidence = confidence
                        racket_bbox = bbox
                        
                        # Calculate center position
                        x_center = (bbox[0] + bbox[2]) / 2
                        y_center = (bbox[1] + bbox[3]) / 2
                        racket_position = np.array([x_center, y_center])
                        
                        # Estimate orientation from bounding box
                        racket_orientation = np.arctan2(height, width)
                        
                        break
        
        # Update tracking if racket detected
        if racket_detected and racket_position is not None:
            self.racket_trajectory.append(racket_position)
            
            # Calculate swing metrics
            if len(self.racket_trajectory) >= 2:
                prev_pos = self.racket_trajectory[-2]
                velocity = racket_position - prev_pos
                self.swing_speed = np.linalg.norm(velocity)
                
                if len(self.racket_trajectory) >= 3:
                    prev_vel = velocity
                    velocity = racket_position - prev_pos
                    self.swing_acceleration = np.linalg.norm(velocity - prev_vel)
            
            self.racket_angle = racket_orientation
        
        return {
            'detected': racket_detected,
            'position': racket_position,
            'confidence': racket_confidence,
            'bbox': racket_bbox,
            'orientation': self.racket_angle,
            'swing_speed': self.swing_speed,
            'swing_acceleration': self.swing_acceleration,
            'trajectory': list(self.racket_trajectory)
        }
    
    def analyze_swing(self, pose_data: Optional[Dict] = None) -> Dict:
        """Analyze swing mechanics combining racket and pose data"""
        
        swing_analysis = {
            'swing_type': 'unknown',
            'swing_quality': 0.0,
            'contact_point': None,
            'follow_through': False,
            'recommendations': []
        }
        
        if len(self.racket_trajectory) < 3:
            return swing_analysis
        
        trajectory = np.array(self.racket_trajectory)
        
        # Analyze swing arc
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        
        # Calculate swing path curvature
        if len(trajectory) >= 3:
            # Fit polynomial to swing path
            coeffs = np.polyfit(x_coords, y_coords, 2)
            curvature = abs(coeffs[0]) * 100  # Scale for readability
            
            # Determine swing type based on motion pattern
            y_range = np.max(y_coords) - np.min(y_coords)
            x_range = np.max(x_coords) - np.min(x_coords)
            
            if y_range > x_range * 1.5:
                swing_analysis['swing_type'] = 'topspin'
            elif x_range > y_range * 1.5:
                swing_analysis['swing_type'] = 'slice'
            else:
                swing_analysis['swing_type'] = 'flat'
            
            # Swing quality based on smoothness
            velocities = np.diff(trajectory, axis=0)
            speed_variance = np.var(np.linalg.norm(velocities, axis=1))
            swing_analysis['swing_quality'] = max(0, 1.0 - speed_variance * 0.1)
        
        # Combine with pose data if available
        if pose_data and 'tennis_metrics' in pose_data:
            metrics = pose_data['tennis_metrics']
            
            # Check for proper follow-through
            if metrics.get('swing_phase') == 'follow_through':
                swing_analysis['follow_through'] = True
            
            # Contact point estimation
            if metrics.get('swing_phase') == 'contact':
                if len(self.racket_trajectory) > 0:
                    swing_analysis['contact_point'] = self.racket_trajectory[-1]
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, swing_analysis)
            swing_analysis['recommendations'] = recommendations
        
        return swing_analysis
    
    def _generate_recommendations(self, pose_metrics: Dict, 
                                 swing_analysis: Dict) -> List[str]:
        """Generate coaching recommendations based on analysis"""
        
        recommendations = []
        
        # Balance recommendations
        balance_score = pose_metrics.get('balance_score', 0)
        if balance_score < 0.7:
            recommendations.append("Improve stance balance - keep shoulders level")
        
        # Body rotation recommendations
        body_rotation = pose_metrics.get('body_rotation', 0)
        if body_rotation < 0.3:
            recommendations.append("Increase hip-shoulder separation for more power")
        
        # Swing quality recommendations
        swing_quality = swing_analysis.get('swing_quality', 0)
        if swing_quality < 0.6:
            recommendations.append("Focus on smoother swing acceleration")
        
        # Follow-through recommendations
        if not swing_analysis.get('follow_through', False):
            recommendations.append("Complete your follow-through for better control")
        
        # Swing type specific advice
        swing_type = swing_analysis.get('swing_type', 'unknown')
        if swing_type == 'slice' and swing_analysis.get('swing_quality', 0) < 0.7:
            recommendations.append("Keep racket face more open for slice shots")
        
        return recommendations
    
    def detect_racket_face_angle(self, frame: np.ndarray, 
                                bbox: np.ndarray) -> float:
        """Estimate racket face angle from image"""
        
        if bbox is None:
            return 0.0
        
        # Extract racket region
        x1, y1, x2, y2 = map(int, bbox)
        racket_roi = frame[y1:y2, x1:x2]
        
        if racket_roi.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(racket_roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Average angle of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        
        if angles:
            return float(np.mean(angles))
        
        return 0.0
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.racket_trajectory.clear()
        self.swing_speed = 0.0
        self.swing_acceleration = 0.0
        self.racket_angle = 0.0
        self.swing_phases = []
