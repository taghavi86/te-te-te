"""
Dynamic Time Warping Engine
هم‌ترازی زمانی برای مقایسه ضربات بازیکنان
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DTWAlignment:
    """نتیجه هم‌ترازی DTW"""
    distance: float
    normalized_distance: float
    alignment_path: List[Tuple[int, int]]
    similarity_score: float


class DTWEngine:
    """موتور Dynamic Time Warping برای مقایسه توالی‌های زمانی"""
    
    def __init__(self, config: dict):
        self.config = config
        self.dtw_config = config.get('dtw', {})
        self.window_size = self.dtw_config.get('window', 20)
        self.feature_weights = self.dtw_config.get('feature_weights', {
            'elbow_angle': 1.3,
            'shoulder_rotation': 1.5,
            'hip_rotation': 1.4,
            'wrist_angle': 1.0,
            'knee_angle': 0.9,
            'trunk_angle': 1.2,
            'wrist_velocity': 1.1,
            'elbow_velocity': 1.0,
            'shoulder_velocity': 1.3
        })
    
    def extract_feature_vector(self, frame_features: dict) -> np.ndarray:
        """
        استخراج بردار ویژگی از داده‌های بیومکانیکی یک فریم
        
        Returns:
            بردار ویژگی‌های نرمال‌شده
        """
        features = []
        feature_names = []
        
        # زوایای مفاصل
        angles = frame_features.get('angles', [])
        angle_dict = {a.joint: a.value for a in angles}
        
        for joint in ['elbow_l', 'elbow_r', 'shoulder_l', 'shoulder_r', 
                     'hip_l', 'hip_r', 'knee_l', 'knee_r']:
            if joint in angle_dict:
                features.append(angle_dict[joint] / 180.0)  # نرمال‌سازی به [0, 1]
                feature_names.append(f'{joint}_angle')
            else:
                features.append(0.0)
                feature_names.append(f'{joint}_angle')
        
        # سرعت‌ها
        velocities = frame_features.get('velocities', {})
        for kp in ['wrist_l', 'wrist_r', 'elbow_l', 'elbow_r', 'shoulder_l', 'shoulder_r']:
            if kp in velocities:
                speed = velocities[kp].value / 100.0  # نرمال‌سازی تقریبی
                features.append(min(speed, 1.0))
                feature_names.append(f'{kp}_velocity')
            else:
                features.append(0.0)
                feature_names.append(f'{kp}_velocity')
        
        # جهت تنه
        orientations = frame_features.get('orientations', [])
        trunk_orientation = next(
            (o.angle for o in orientations if o.start_point == 'shoulder_l' and o.end_point == 'hip_l'),
            0.0
        )
        features.append(trunk_orientation / 180.0)
        feature_names.append('trunk_angle')
        
        return np.array(features), feature_names
    
    def calculate_distance(self, seq1: List[np.ndarray], 
                          seq2: List[np.ndarray]) -> np.ndarray:
        """
        محاسبه ماتریس فاصله بین دو توالی
        
        Args:
            seq1: توالی اول (لیست بردارهای ویژگی)
            seq2: توالی دوم
            
        Returns:
            ماتریس فاصله
        """
        n = len(seq1)
        m = len(seq2)
        
        # ماتریس فاصله
        dist_matrix = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                # فاصله وزنی اقلیدسی
                diff = seq1[i] - seq2[j]
                
                # اعمال وزن‌ها
                weights = np.array([
                    self.feature_weights.get(f'feature_{k}', 1.0) 
                    for k in range(len(diff))
                ])
                
                weighted_diff = diff * np.sqrt(weights)
                dist_matrix[i, j] = np.sqrt(np.sum(weighted_diff ** 2))
        
        return dist_matrix
    
    def compute_dtw(self, sequence_user: List[dict],
                   sequence_pro: List[dict]) -> DTWAlignment:
        """
        محاسبه DTW بین دو توالی از فریم‌ها
        
        Args:
            sequence_user: لیست ویژگی‌های بیومکانیکی کاربر
            sequence_pro: لیست ویژگی‌های بیومکانیکی بازیکن حرفه‌ای
            
        Returns:
            نتیجه هم‌ترازی شامل فاصله، مسیر و امتیاز شباهت
        """
        # استخراج بردارهای ویژگی
        user_vectors = []
        pro_vectors = []
        
        for frame_data in sequence_user:
            vec, _ = self.extract_feature_vector(frame_data)
            user_vectors.append(vec)
        
        for frame_data in sequence_pro:
            vec, _ = self.extract_feature_vector(frame_data)
            pro_vectors.append(vec)
        
        n = len(user_vectors)
        m = len(pro_vectors)
        
        if n == 0 or m == 0:
            return DTWAlignment(
                distance=float('inf'),
                normalized_distance=1.0,
                alignment_path=[],
                similarity_score=0.0
            )
        
        # محاسبه ماتریس فاصله
        dist_matrix = self.calculate_distance(user_vectors, pro_vectors)
        
        # ماتریس هزینه تجمعی
        cost_matrix = np.full((n, m), float('inf'))
        cost_matrix[0, 0] = dist_matrix[0, 0]
        
        # پر کردن سطر اول
        for j in range(1, m):
            cost_matrix[0, j] = cost_matrix[0, j-1] + dist_matrix[0, j]
        
        # پر کردن ستون اول
        for i in range(1, n):
            cost_matrix[i, 0] = cost_matrix[i-1, 0] + dist_matrix[i, 0]
        
        # پر کردن بقیه ماتریس با محدودیت پنجره
        for i in range(1, n):
            for j in range(1, m):
                # محدودیت Sakoe-Chiba band
                if abs(i - j) > self.window_size:
                    continue
                
                min_prev = min(
                    cost_matrix[i-1, j],    # Insertion
                    cost_matrix[i, j-1],    # Deletion
                    cost_matrix[i-1, j-1]   # Match
                )
                
                cost_matrix[i, j] = dist_matrix[i, j] + min_prev
        
        # فاصله DTW نهایی
        dtw_distance = cost_matrix[n-1, m-1]
        
        # نرمال‌سازی فاصله
        max_possible_dist = (n + m) * np.max(dist_matrix) if np.max(dist_matrix) > 0 else 1
        normalized_distance = dtw_distance / max_possible_dist
        
        # بازسازی مسیر بهینه
        alignment_path = self._backtrack(cost_matrix, n, m)
        
        # محاسبه امتیاز شباهت
        similarity_score = max(0, 1.0 - normalized_distance)
        
        return DTWAlignment(
            distance=dtw_distance,
            normalized_distance=normalized_distance,
            alignment_path=alignment_path,
            similarity_score=similarity_score
        )
    
    def _backtrack(self, cost_matrix: np.ndarray, n: int, m: int) -> List[Tuple[int, int]]:
        """بازسازی مسیر بهینه از ماتریس هزینه"""
        path = []
        i, j = n - 1, m - 1
        
        path.append((i, j))
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # انتخاب کمترین هزینه
                candidates = [
                    (i-1, j-1, cost_matrix[i-1, j-1]),
                    (i-1, j, cost_matrix[i-1, j]),
                    (i, j-1, cost_matrix[i, j-1])
                ]
                
                # پیدا کردن کمترین
                min_candidate = min(candidates, key=lambda x: x[2])
                i, j = min_candidate[0], min_candidate[1]
            
            path.append((i, j))
        
        path.reverse()
        return path
    
    def compute_feature_alignment(self, sequence_user: List[dict],
                                  sequence_pro: List[dict],
                                  feature_name: str) -> Dict:
        """
        محاسبه هم‌ترازی برای یک ویژگی خاص
        
        Returns:
            دیکشنری شامل مقادیر ویژگی در هر نقطه از مسیر هم‌ترازی
        """
        dtw_result = self.compute_dtw(sequence_user, sequence_pro)
        
        if not dtw_result.alignment_path:
            return {'aligned_values': [], 'differences': []}
        
        aligned_data = []
        
        for user_idx, pro_idx in dtw_result.alignment_path:
            if user_idx < len(sequence_user) and pro_idx < len(sequence_pro):
                user_frame = sequence_user[user_idx]
                pro_frame = sequence_pro[pro_idx]
                
                # استخراج ویژگی مورد نظر
                user_val = self._extract_single_feature(user_frame, feature_name)
                pro_val = self._extract_single_feature(pro_frame, feature_name)
                
                aligned_data.append({
                    'user_frame': user_idx,
                    'pro_frame': pro_idx,
                    'user_value': user_val,
                    'pro_value': pro_val,
                    'difference': user_val - pro_val if user_val is not None and pro_val is not None else None
                })
        
        return {
            'aligned_values': aligned_data,
            'dtw_distance': dtw_result.distance,
            'similarity_score': dtw_result.similarity_score
        }
    
    def _extract_single_feature(self, frame_data: dict, feature_name: str) -> Optional[float]:
        """استخراج یک ویژگی خاص از داده‌های فریم"""
        if '_angle' in feature_name:
            joint = feature_name.replace('_angle', '')
            angles = frame_data.get('angles', [])
            for angle in angles:
                if angle.joint == joint:
                    return angle.value
            return None
        
        elif '_velocity' in feature_name:
            kp = feature_name.replace('_velocity', '')
            velocities = frame_data.get('velocities', {})
            if kp in velocities:
                return velocities[kp].value
            return None
        
        elif feature_name == 'trunk_angle':
            orientations = frame_data.get('orientations', [])
            for orient in orientations:
                if orient.start_point == 'shoulder_l' and orient.end_point == 'hip_l':
                    return orient.angle
            return None
        
        return None
    
    def compute_multivariate_dtw(self, sequences_user: List[List[dict]],
                                sequences_pro: List[List[dict]]) -> Dict:
        """
        محاسبه DTW چندمتغیره برای چندین ضربه
        
        Returns:
            میانگین فواصل و شباهت‌ها
        """
        if not sequences_user or not sequences_pro:
            return {
                'avg_distance': float('inf'),
                'avg_similarity': 0.0,
                'stroke_count': 0
            }
        
        # جفت کردن ضربات (ساده‌ترین حالت: جفت کردن بر اساس ایندکس)
        distances = []
        similarities = []
        
        min_strokes = min(len(sequences_user), len(sequences_pro))
        
        for i in range(min_strokes):
            result = self.compute_dtw(sequences_user[i], sequences_pro[i])
            distances.append(result.distance)
            similarities.append(result.similarity_score)
        
        return {
            'avg_distance': np.mean(distances) if distances else float('inf'),
            'avg_similarity': np.mean(similarities) if similarities else 0.0,
            'stroke_count': min_strokes,
            'individual_distances': distances,
            'individual_similarities': similarities
        }
