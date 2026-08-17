"""
Biomechanics Engine - محاسبه ویژگی‌های بیومکانیکی از keypoints
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import euclidean


@dataclass
class JointAngle:
    """زاویه مفصل"""
    joint: str
    value: float  # degrees
    confidence: float
    frame: int


@dataclass
class SegmentOrientation:
    """جهت segment بدن"""
    start_point: str
    end_point: str
    angle: float  # degrees relative to horizontal
    length: float
    confidence: float


@dataclass
class Velocity:
    """سرعت خطی"""
    keypoint: str
    value: float  # pixels/frame
    vector: np.ndarray
    confidence: float


@dataclass
class AngularVelocity:
    """سرعت زاویه‌ای"""
    joint: str
    value: float  # degrees/frame
    acceleration: float  # degrees/frame²
    confidence: float


class BiomechanicsEngine:
    """موتور محاسبات بیومکانیکی"""
    
    def __init__(self, config: dict):
        self.config = config
        self.body_segments = {
            'upper_arm_l': ('shoulder_l', 'elbow_l'),
            'lower_arm_l': ('elbow_l', 'wrist_l'),
            'upper_arm_r': ('shoulder_r', 'elbow_r'),
            'lower_arm_r': ('elbow_r', 'wrist_r'),
            'upper_leg_l': ('hip_l', 'knee_l'),
            'lower_leg_l': ('knee_l', 'ankle_l'),
            'upper_leg_r': ('hip_r', 'knee_r'),
            'lower_leg_r': ('knee_r', 'ankle_r'),
            'torso': ('shoulder_l', 'hip_l'),
            'shoulders': ('shoulder_l', 'shoulder_r'),
            'hips': ('hip_l', 'hip_r'),
        }
        
        self.joint_triplets = {
            'elbow_l': ('shoulder_l', 'elbow_l', 'wrist_l'),
            'elbow_r': ('shoulder_r', 'elbow_r', 'wrist_r'),
            'knee_l': ('hip_l', 'knee_l', 'ankle_l'),
            'knee_r': ('hip_r', 'knee_r', 'ankle_r'),
            'shoulder_l': ('hip_l', 'shoulder_l', 'elbow_l'),
            'shoulder_r': ('hip_r', 'shoulder_r', 'elbow_r'),
            'hip_l': ('shoulder_l', 'hip_l', 'knee_l'),
            'hip_r': ('shoulder_r', 'hip_r', 'knee_r'),
        }
    
    def calculate_angle(self, point_a: np.ndarray, point_b: np.ndarray, 
                       point_c: np.ndarray) -> float:
        """
        محاسبه زاویه بین سه نقطه (A-B-C)
        B نقطه وسط (مفصل) است
        """
        if any(p is None for p in [point_a, point_b, point_c]):
            return None
        
        # Vector BA
        ba = point_a - point_b
        # Vector BC
        bc = point_c - point_b
        
        # Normalize
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return None
        
        ba = ba / ba_norm
        bc = bc / bc_norm
        
        # Dot product
        dot = np.clip(np.dot(ba, bc), -1.0, 1.0)
        
        # Angle in degrees
        angle = np.degrees(np.arccos(dot))
        
        return angle
    
    def calculate_joint_angles(self, keypoints: Dict[str, np.ndarray], 
                               confidences: Dict[str, float],
                               frame: int) -> List[JointAngle]:
        """محاسبه تمام زوایای مفاصل"""
        angles = []
        
        for joint, (p1, p2, p3) in self.joint_triplets.items():
            if all(k in keypoints for k in [p1, p2, p3]):
                conf = min(confidences.get(p1, 0), 
                          confidences.get(p2, 0), 
                          confidences.get(p3, 0))
                
                if conf < self.config['pose']['confidence_threshold']:
                    continue
                
                angle = self.calculate_angle(
                    keypoints[p1], 
                    keypoints[p2], 
                    keypoints[p3]
                )
                
                if angle is not None:
                    angles.append(JointAngle(
                        joint=joint,
                        value=angle,
                        confidence=conf,
                        frame=frame
                    ))
        
        return angles
    
    def calculate_segment_orientation(self, keypoints: Dict[str, np.ndarray],
                                     confidences: Dict[str, float]) -> List[SegmentOrientation]:
        """محاسبه جهت segmentهای بدن"""
        orientations = []
        
        for seg_name, (start, end) in self.body_segments.items():
            if start not in keypoints or end not in keypoints:
                continue
            
            conf = min(confidences.get(start, 0), confidences.get(end, 0))
            
            if conf < self.config['pose']['confidence_threshold']:
                continue
            
            start_pt = keypoints[start]
            end_pt = keypoints[end]
            
            # Vector
            vec = end_pt - start_pt
            
            # Length
            length = np.linalg.norm(vec)
            
            # Angle relative to horizontal
            angle = np.degrees(np.arctan2(vec[1], vec[0]))
            
            orientations.append(SegmentOrientation(
                start_point=start,
                end_point=end,
                angle=angle,
                length=length,
                confidence=conf
            ))
        
        return orientations
    
    def calculate_velocity(self, current_pos: np.ndarray, 
                          previous_pos: np.ndarray,
                          dt: float) -> Optional[Velocity]:
        """محاسبه سرعت خطی"""
        if current_pos is None or previous_pos is None:
            return None
        
        diff = current_pos - previous_pos
        speed = np.linalg.norm(diff) / dt if dt > 0 else 0
        
        return Velocity(
            keypoint='unknown',
            value=speed,
            vector=diff / dt if dt > 0 else np.zeros_like(diff),
            confidence=1.0
        )
    
    def calculate_angular_velocity(self, current_angle: float,
                                   previous_angle: float,
                                   current_time: float,
                                   previous_time: float) -> Optional[AngularVelocity]:
        """محاسبه سرعت زاویه‌ای و شتاب زاویه‌ای"""
        dt = current_time - previous_time
        
        if dt <= 0 or previous_angle is None:
            return None
        
        # Angular velocity
        omega = (current_angle - previous_angle) / dt
        
        # For acceleration, we need more history (simplified here)
        alpha = 0.0  # Would need previous omega for accurate calculation
        
        return AngularVelocity(
            joint='unknown',
            value=omega,
            acceleration=alpha,
            confidence=1.0
        )
    
    def calculate_motion_energy(self, velocities: List[np.ndarray]) -> float:
        """
        محاسبه انرژی حرکتی
        E(t) = Σ velocity²
        """
        if not velocities:
            return 0.0
        
        total_energy = sum(np.sum(v ** 2) for v in velocities if v is not None)
        return total_energy
    
    def calculate_body_relative_coordinates(self, keypoints: Dict[str, np.ndarray],
                                           reference_point: str = 'shoulder_l') -> Dict[str, np.ndarray]:
        """
        تبدیل مختصات به سیستم مختصات نسبی بدن
        """
        if reference_point not in keypoints:
            return {}
        
        ref = keypoints[reference_point]
        relative = {}
        
        for name, pos in keypoints.items():
            if pos is not None:
                relative[name] = pos - ref
        
        return relative
    
    def normalize_by_body_scale(self, keypoints: Dict[str, np.ndarray],
                               scale_reference: str = 'shoulder_width') -> Dict[str, np.ndarray]:
        """
        نرمال‌سازی مختصات بر اساس مقیاس بدن
        """
        if scale_reference == 'shoulder_width':
            if 'shoulder_l' not in keypoints or 'shoulder_r' not in keypoints:
                return keypoints
            
            shoulder_l = keypoints['shoulder_l']
            shoulder_r = keypoints['shoulder_r']
            
            scale = np.linalg.norm(shoulder_r - shoulder_l)
            
            if scale < 1e-6:
                return keypoints
            
            normalized = {}
            for name, pos in keypoints.items():
                if pos is not None:
                    normalized[name] = (pos - shoulder_l) / scale
            
            return normalized
        
        return keypoints
    
    def extract_frame_features(self, keypoints: Dict[str, np.ndarray],
                              confidences: Dict[str, float],
                              frame: int,
                              timestamp: float,
                              previous_data: Optional[dict] = None) -> dict:
        """
        استخراج تمام ویژگی‌های بیومکانیکی برای یک فریم
        """
        features = {
            'frame': frame,
            'timestamp': timestamp,
            'angles': [],
            'orientations': [],
            'velocities': {},
            'angular_velocities': {},
            'motion_energy': 0.0,
            'keypoints': keypoints.copy(),
            'confidences': confidences.copy()
        }
        
        # Calculate joint angles
        features['angles'] = self.calculate_joint_angles(keypoints, confidences, frame)
        
        # Calculate segment orientations
        features['orientations'] = self.calculate_segment_orientation(keypoints, confidences)
        
        # Calculate velocities if previous data available
        if previous_data and 'keypoints' in previous_data:
            velocities = []
            for kp_name in keypoints.keys():
                if kp_name in previous_data['keypoints']:
                    current_pos = keypoints[kp_name]
                    prev_pos = previous_data['keypoints'][kp_name]
                    
                    if current_pos is not None and prev_pos is not None:
                        dt = timestamp - previous_data['timestamp']
                        vel = self.calculate_velocity(current_pos, prev_pos, dt)
                        
                        if vel:
                            vel.keypoint = kp_name
                            features['velocities'][kp_name] = vel
                            velocities.append(vel.vector)
            
            # Motion energy
            features['motion_energy'] = self.calculate_motion_energy(velocities)
        
        # Calculate angular velocities for angles
        if previous_data and 'angles' in previous_data:
            for angle in features['angles']:
                prev_angle_data = next(
                    (a for a in previous_data['angles'] if a.joint == angle.joint),
                    None
                )
                
                if prev_angle_data:
                    dt = timestamp - previous_data['timestamp']
                    ang_vel = self.calculate_angular_velocity(
                        angle.value,
                        prev_angle_data.value,
                        timestamp,
                        previous_data['timestamp']
                    )
                    
                    if ang_vel:
                        ang_vel.joint = angle.joint
                        features['angular_velocities'][angle.joint] = ang_vel
        
        return features
