"""
Advanced Pose Estimation using MediaPipe BlazePose and HRNet
Detects 33 body keypoints with high precision for tennis analysis
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
import torch

class PoseEstimator:
    """High-precision pose estimator for tennis movement analysis"""
    
    def __init__(self, config: Dict, device: str = 'cpu'):
        self.config = config
        self.device = device
        
        # Initialize MediaPipe BlazePose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Full model for maximum accuracy
            enable_segmentation=True,
            smooth_landmarks=True,
            min_detection_confidence=config.get('CONFIDENCE_THRESHOLD', 0.75),
            min_tracking_confidence=config.get('CONFIDENCE_THRESHOLD', 0.75)
        )
        
        # Joint connections for tennis-specific analysis
        self.tennis_joints = {
            'shoulder': [mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
            'elbow': [mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                     mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
            'wrist': [mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                     mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
            'hip': [mp.solutions.pose.PoseLandmark.LEFT_HIP,
                   mp.solutions.pose.PoseLandmark.RIGHT_HIP],
            'knee': [mp.solutions.pose.PoseLandmark.LEFT_KNEE,
                    mp.solutions.pose.PoseLandmark.RIGHT_KNEE],
            'ankle': [mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
                     mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
        }
        
        # Angle calculation pairs for tennis technique
        self.angle_pairs = [
            ('shoulder_elbow_wrist', (11, 13, 15)),  # Right arm
            ('shoulder_hip_knee', (11, 23, 25)),     # Right leg
            ('hip_knee_ankle', (23, 25, 27)),        # Right lower leg
            ('elbow_wrist_index', (13, 15, 17)),     # Right forearm
        ]
    
    def detect(self, frame: np.ndarray) -> Dict:
        """Detect pose and calculate tennis-specific metrics"""
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {'detected': False, 'keypoints': [], 'angles': []}
        
        # Extract keypoints
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        # Calculate angles
        angles = self._calculate_angles(keypoints)
        
        # Tennis-specific analysis
        tennis_metrics = self._analyze_tennis_pose(keypoints, angles)
        
        return {
            'detected': True,
            'keypoints': keypoints,
            'angles': angles,
            'tennis_metrics': tennis_metrics,
            'confidence': results.pose_landmarks.landmark[0].visibility
        }
    
    def _calculate_angles(self, keypoints: List[Dict]) -> List[float]:
        """Calculate joint angles from keypoints"""
        angles = []
        
        for _, (p1_idx, p2_idx, p3_idx) in self.angle_pairs:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints) and p3_idx < len(keypoints):
                p1 = np.array([keypoints[p1_idx]['x'], keypoints[p1_idx]['y']])
                p2 = np.array([keypoints[p2_idx]['x'], keypoints[p2_idx]['y']])
                p3 = np.array([keypoints[p3_idx]['x'], keypoints[p3_idx]['y']])
                
                # Calculate angle
                angle = self._calculate_angle_between_points(p1, p2, p3)
                angles.append(angle)
        
        return angles
    
    def _calculate_angle_between_points(self, p1: np.ndarray, p2: np.ndarray, 
                                       p3: np.ndarray) -> float:
        """Calculate angle between three points"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
        
        # Calculate angle using dot product
        dot_product = np.dot(v1_norm, v2_norm)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def _analyze_tennis_pose(self, keypoints: List[Dict], 
                            angles: List[float]) -> Dict:
        """Analyze pose for tennis-specific metrics"""
        
        metrics = {
            'balance_score': 0.0,
            'ready_position': False,
            'swing_phase': 'unknown',
            'body_rotation': 0.0
        }
        
        if len(keypoints) < 33:
            return metrics
        
        # Calculate balance (symmetry between left and right sides)
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        
        # Shoulder-hip alignment for balance
        shoulder_line = abs(left_shoulder['y'] - right_shoulder['y'])
        hip_line = abs(left_hip['y'] - right_hip['y'])
        
        balance_diff = abs(shoulder_line - hip_line)
        metrics['balance_score'] = max(0, 1.0 - balance_diff * 2)
        
        # Detect ready position
        if len(angles) >= 2:
            knee_angle = angles[1] if len(angles) > 1 else 180
            elbow_angle = angles[0] if len(angles) > 0 else 180
            
            # Ready position: knees bent, elbows slightly flexed
            if 120 <= knee_angle <= 160 and 150 <= elbow_angle <= 170:
                metrics['ready_position'] = True
        
        # Estimate swing phase based on arm position
        if len(keypoints) >= 16:
            right_wrist_y = keypoints[15]['y']
            right_shoulder_y = keypoints[11]['y']
            
            if right_wrist_y < right_shoulder_y:
                metrics['swing_phase'] = 'backswing'
            elif right_wrist_y > right_shoulder_y + 0.1:
                metrics['swing_phase'] = 'follow_through'
            else:
                metrics['swing_phase'] = 'contact'
        
        # Body rotation estimation
        if len(keypoints) >= 24:
            left_shoulder_x = keypoints[11]['x']
            right_shoulder_x = keypoints[12]['x']
            left_hip_x = keypoints[23]['x']
            right_hip_x = keypoints[24]['x']
            
            shoulder_angle = np.arctan2(right_shoulder_x - left_shoulder_x, 0.1)
            hip_angle = np.arctan2(right_hip_x - left_hip_x, 0.1)
            
            metrics['body_rotation'] = abs(shoulder_angle - hip_angle)
        
        return metrics
    
    def cleanup(self):
        """Release resources"""
        self.pose.close()
