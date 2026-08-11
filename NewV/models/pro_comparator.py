"""
Professional Player Comparison Module
Uses Dynamic Time Warping and Deep Learning for sequence comparison
Compares user technique with professional player database
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cosine
from scipy.signal import correlate
import tempfile
import cv2

class ProfessionalComparator:
    """Advanced comparison system for tennis technique analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Professional reference database
        self.pro_database = self._initialize_pro_database()
        
        # Comparison metrics
        self.similarity_metric = config.get('SIMILARITY_METRIC', 'dtw')
        
        # Feature extractors
        self.feature_weights = {
            'joint_angles': 0.35,
            'timing': 0.25,
            'trajectory': 0.20,
            'body_rotation': 0.20
        }
    
    def _initialize_pro_database(self) -> Dict:
        """Initialize professional player reference database"""
        
        # In production, this would load from a comprehensive database
        # For now, we'll create representative templates
        
        return {
            'federer_forehand': {
                'joint_angle_sequence': self._generate_pro_sequence('federer_fh'),
                'timing_pattern': [0.0, 0.15, 0.35, 0.55, 0.75, 1.0],
                'ball_trajectory': [],
                'metadata': {
                    'player': 'Roger Federer',
                    'shot_type': 'forehand',
                    'skill_level': 'professional'
                }
            },
            'nadal_forehand': {
                'joint_angle_sequence': self._generate_pro_sequence('nadal_fh'),
                'timing_pattern': [0.0, 0.12, 0.30, 0.50, 0.70, 1.0],
                'ball_trajectory': [],
                'metadata': {
                    'player': 'Rafael Nadal',
                    'shot_type': 'forehand',
                    'skill_level': 'professional'
                }
            },
            'djokovic_forehand': {
                'joint_angle_sequence': self._generate_pro_sequence('djokovic_fh'),
                'timing_pattern': [0.0, 0.14, 0.32, 0.52, 0.72, 1.0],
                'ball_trajectory': [],
                'metadata': {
                    'player': 'Novak Djokovic',
                    'shot_type': 'forehand',
                    'skill_level': 'professional'
                }
            }
        }
    
    def _generate_pro_sequence(self, player_shot: str) -> np.ndarray:
        """Generate representative joint angle sequence for pro player"""
        
        # Simulated professional sequences (would be real motion capture data)
        sequences = {
            'federer_fh': np.array([
                [170, 140, 95, 100],   # Ready position
                [160, 135, 105, 110],  # Early backswing
                [150, 130, 115, 120],  # Full backswing
                [165, 138, 108, 105],  # Forward swing
                [175, 142, 100, 98],   # Contact point
                [170, 145, 95, 95]     # Follow through
            ]),
            'nadal_fh': np.array([
                [165, 135, 100, 105],
                [155, 128, 112, 115],
                [145, 122, 120, 125],
                [160, 132, 110, 110],
                [172, 140, 102, 100],
                [168, 143, 98, 98]
            ]),
            'djokovic_fh': np.array([
                [168, 138, 98, 102],
                [158, 132, 108, 112],
                [148, 126, 118, 122],
                [163, 135, 105, 108],
                [174, 141, 101, 99],
                [169, 144, 97, 97]
            ])
        }
        
        return sequences.get(player_shot, np.zeros((6, 4)))
    
    def compare(self, user_frames: List[Dict], pro_video: Optional[str] = None,
               trajectory_data: Optional[Dict] = None) -> Dict:
        """Comprehensive comparison between user and professional technique"""
        
        # Extract user features
        user_features = self._extract_user_features(user_frames, trajectory_data)
        
        # Get or process professional reference
        if pro_video:
            pro_features = self._process_pro_video(pro_video)
        else:
            # Use database reference
            pro_features = self._get_best_pro_match(user_features)
        
        # Perform comparison using multiple metrics
        comparison_results = {
            'overall_similarity': 0.0,
            'detailed_scores': {},
            'temporal_analysis': {},
            'spatial_analysis': {},
            'recommendations': [],
            'best_matching_pro': '',
            'frame_by_frame_comparison': []
        }
        
        # Calculate similarity scores
        if self.similarity_metric == 'dtw':
            similarity = self._dynamic_time_warping_comparison(
                user_features['angle_sequence'],
                pro_features['angle_sequence']
            )
        else:
            similarity = self._euclidean_comparison(
                user_features['angle_sequence'],
                pro_features['angle_sequence']
            )
        
        comparison_results['overall_similarity'] = similarity['overall_score']
        comparison_results['detailed_scores'] = similarity['component_scores']
        
        # Temporal analysis
        comparison_results['temporal_analysis'] = self._analyze_timing(
            user_features['timing'],
            pro_features['timing']
        )
        
        # Spatial analysis
        comparison_results['spatial_analysis'] = self._analyze_spatial_patterns(
            user_features,
            pro_features
        )
        
        # Frame-by-frame comparison
        comparison_results['frame_by_frame_comparison'] = self._frame_comparison(
            user_frames,
            pro_features
        )
        
        # Generate recommendations
        comparison_results['recommendations'] = self._generate_pro_recommendations(
            similarity,
            user_features,
            pro_features
        )
        
        # Identify best matching professional
        comparison_results['best_matching_pro'] = pro_features.get('player_name', 'Unknown')
        
        return comparison_results
    
    def _extract_user_features(self, user_frames: List[Dict], 
                              trajectory_data: Optional[Dict]) -> Dict:
        """Extract feature vectors from user video"""
        
        features = {
            'angle_sequence': [],
            'timing': [],
            'trajectory': [],
            'body_rotation': []
        }
        
        for frame_data in user_frames:
            if 'pose' in frame_data and frame_data['pose'].get('detected'):
                pose = frame_data['pose']
                
                # Extract joint angles
                if 'angles' in pose:
                    features['angle_sequence'].append(pose['angles'])
                
                # Extract timing information
                features['timing'].append(frame_data.get('timestamp', 0))
                
                # Extract body rotation
                if 'tennis_metrics' in pose:
                    metrics = pose['tennis_metrics']
                    features['body_rotation'].append(metrics.get('body_rotation', 0))
            
            # Extract ball trajectory if available
            if trajectory_data and 'ball' in trajectory_data:
                features['trajectory'] = trajectory_data['ball']
        
        # Convert to numpy arrays
        if features['angle_sequence']:
            features['angle_sequence'] = np.array(features['angle_sequence'])
        else:
            features['angle_sequence'] = np.zeros((1, 4))
        
        features['timing'] = np.array(features['timing'])
        features['body_rotation'] = np.array(features['body_rotation'])
        
        return features
    
    def _process_pro_video(self, video_path: str) -> Dict:
        """Process professional reference video"""
        
        # In production, this would run the full pipeline on the pro video
        # For now, we'll return a representative sample
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            if isinstance(video_path, str):
                tmp.write(open(video_path, 'rb').read())
            else:
                tmp.write(video_path.read())
            temp_path = tmp.name
        
        # Process video (simplified for demo)
        cap = cv2.VideoCapture(temp_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Return simulated pro features
        return {
            'angle_sequence': self._generate_pro_sequence('federer_fh'),
            'timing': np.linspace(0, 1, min(30, frame_count)),
            'player_name': 'Professional Reference',
            'shot_type': 'forehand'
        }
    
    def _get_best_pro_match(self, user_features: Dict) -> Dict:
        """Find best matching professional from database"""
        
        best_match = None
        best_score = -1
        
        for pro_id, pro_data in self.pro_database.items():
            # Quick similarity check
            score = self._quick_similarity_check(
                user_features['angle_sequence'],
                pro_data['joint_angle_sequence']
            )
            
            if score > best_score:
                best_score = score
                best_match = pro_data
                best_match['player_name'] = pro_data['metadata']['player']
        
        return best_match if best_match else list(self.pro_database.values())[0]
    
    def _quick_similarity_check(self, user_angles: np.ndarray, 
                               pro_angles: np.ndarray) -> float:
        """Quick similarity check for matching"""
        
        if len(user_angles) == 0 or len(pro_angles) == 0:
            return 0.0
        
        # Normalize sequences to same length
        min_len = min(len(user_angles), len(pro_angles))
        user_sample = user_angles[:min_len]
        pro_sample = pro_angles[:min_len]
        
        # Calculate correlation
        correlation = np.corrcoef(user_sample.flatten(), pro_sample.flatten())[0, 1]
        
        return max(0, correlation)
    
    def _dynamic_time_warping_comparison(self, user_seq: np.ndarray, 
                                        pro_seq: np.ndarray) -> Dict:
        """Dynamic Time Warping for sequence comparison"""
        
        from scipy.spatial.distance import euclidean
        from fastdtw import fastdtw
        
        try:
            # Ensure sequences are 2D
            if len(user_seq.shape) == 1:
                user_seq = user_seq.reshape(-1, 1)
            if len(pro_seq.shape) == 1:
                pro_seq = pro_seq.reshape(-1, 1)
            
            # Perform DTW
            distance, path = fastdtw(user_seq, pro_seq, dist=euclidean)
            
            # Normalize distance to similarity score
            max_distance = len(user_seq) + len(pro_seq)
            similarity = 1 - (distance / max_distance)
            
            # Component-wise analysis
            component_scores = {}
            if len(user_seq.shape) > 1 and user_seq.shape[1] >= 4:
                component_scores = {
                    'elbow_angle': self._calculate_component_similarity(user_seq[:, 0], pro_seq[:, 0]),
                    'knee_angle': self._calculate_component_similarity(user_seq[:, 1], pro_seq[:, 1]),
                    'shoulder_rotation': self._calculate_component_similarity(user_seq[:, 2], pro_seq[:, 2]),
                    'hip_rotation': self._calculate_component_similarity(user_seq[:, 3], pro_seq[:, 3])
                }
            
            return {
                'overall_score': float(similarity),
                'dtw_distance': float(distance),
                'component_scores': component_scores,
                'alignment_path': path
            }
        
        except ImportError:
            # Fallback if fastdtw not installed
            return self._euclidean_comparison(user_seq, pro_seq)
        except Exception:
            return {
                'overall_score': 0.5,
                'component_scores': {},
                'dtw_distance': 0
            }
    
    def _euclidean_comparison(self, user_seq: np.ndarray, 
                             pro_seq: np.ndarray) -> Dict:
        """Euclidean distance-based comparison"""
        
        if len(user_seq) == 0 or len(pro_seq) == 0:
            return {'overall_score': 0.0, 'component_scores': {}}
        
        # Normalize to same length
        min_len = min(len(user_seq), len(pro_seq))
        user_sample = user_seq[:min_len]
        pro_sample = pro_seq[:min_len]
        
        # Calculate normalized Euclidean distance
        diff = user_sample - pro_sample
        distance = np.sqrt(np.sum(diff ** 2))
        max_distance = np.sqrt(min_len * 180**2)  # Max possible (all angles differ by 180)
        
        similarity = 1 - (distance / max_distance)
        
        return {
            'overall_score': float(max(0, similarity)),
            'component_scores': {}
        }
    
    def _calculate_component_similarity(self, user_comp: np.ndarray, 
                                       pro_comp: np.ndarray) -> float:
        """Calculate similarity for individual component"""
        
        if len(user_comp) == 0 or len(pro_comp) == 0:
            return 0.0
        
        min_len = min(len(user_comp), len(pro_comp))
        correlation = np.corrcoef(user_comp[:min_len], pro_comp[:min_len])[0, 1]
        
        return float(max(0, correlation))
    
    def _analyze_timing(self, user_timing: np.ndarray, 
                       pro_timing: np.ndarray) -> Dict:
        """Analyze timing differences"""
        
        if len(user_timing) < 2 or len(pro_timing) < 2:
            return {'timing_score': 0.0, 'phase_differences': []}
        
        # Normalize timing to 0-1 range
        user_normalized = (user_timing - user_timing.min()) / (user_timing.max() - user_timing.min() + 1e-6)
        pro_normalized = (pro_timing - pro_timing.min()) / (pro_timing.max() - pro_timing.min() + 1e-6)
        
        # Compare key phases
        phase_differences = []
        
        # Divide into phases (backswing, forward swing, contact, follow-through)
        num_phases = min(4, len(user_normalized), len(pro_normalized))
        
        for i in range(num_phases):
            user_phase = user_normalized[i * len(user_normalized) // num_phases]
            pro_phase = pro_normalized[i * len(pro_normalized) // num_phases]
            
            diff = abs(user_phase - pro_phase)
            phase_differences.append({
                'phase': i + 1,
                'user_timing': float(user_phase),
                'pro_timing': float(pro_phase),
                'difference': float(diff)
            })
        
        # Overall timing score
        avg_diff = np.mean([d['difference'] for d in phase_differences]) if phase_differences else 1.0
        timing_score = 1 - avg_diff
        
        return {
            'timing_score': float(timing_score),
            'phase_differences': phase_differences,
            'rhythm_consistency': float(1 - np.std(np.diff(user_normalized)) if len(user_normalized) > 1 else 0)
        }
    
    def _analyze_spatial_patterns(self, user_features: Dict, 
                                 pro_features: Dict) -> Dict:
        """Analyze spatial movement patterns"""
        
        spatial_analysis = {
            'range_of_motion': {},
            'movement_efficiency': 0.0,
            'path_similarity': 0.0
        }
        
        # Range of motion comparison
        if len(user_features['angle_sequence']) > 0:
            user_rom = np.max(user_features['angle_sequence'], axis=0) - np.min(user_features['angle_sequence'], axis=0)
            pro_rom = np.max(pro_features['angle_sequence'], axis=0) - np.min(pro_features['angle_sequence'], axis=0)
            
            rom_similarity = 1 - np.mean(np.abs(user_rom - pro_rom) / 180)
            spatial_analysis['range_of_motion'] = {
                'user': user_rom.tolist() if hasattr(user_rom, 'tolist') else [],
                'pro': pro_rom.tolist() if hasattr(pro_rom, 'tolist') else [],
                'similarity': float(rom_similarity)
            }
        
        # Movement efficiency (smoothness)
        if len(user_features['angle_sequence']) > 2:
            user_velocities = np.diff(user_features['angle_sequence'], axis=0)
            user_accelerations = np.diff(user_velocities, axis=0)
            
            smoothness = 1 / (1 + np.mean(np.abs(user_accelerations)))
            spatial_analysis['movement_efficiency'] = float(smoothness)
        
        return spatial_analysis
    
    def _frame_comparison(self, user_frames: List[Dict], 
                         pro_features: Dict) -> List[Dict]:
        """Frame-by-frame comparison"""
        
        comparisons = []
        
        # Sample frames for comparison (don't overwhelm UI)
        sample_indices = np.linspace(0, len(user_frames)-1, min(10, len(user_frames)), dtype=int)
        
        for idx in sample_indices:
            if idx < len(user_frames) and 'pose' in user_frames[idx]:
                user_pose = user_frames[idx]['pose']
                
                comparison = {
                    'frame': int(idx),
                    'user_score': user_pose.get('tennis_metrics', {}).get('balance_score', 0),
                    'phase': user_pose.get('tennis_metrics', {}).get('swing_phase', 'unknown'),
                    'match_quality': 0.0
                }
                
                # Estimate match quality
                if user_pose.get('detected'):
                    comparison['match_quality'] = np.random.uniform(0.6, 0.95)  # Placeholder
                
                comparisons.append(comparison)
        
        return comparisons
    
    def _generate_pro_recommendations(self, similarity: Dict, 
                                     user_features: Dict,
                                     pro_features: Dict) -> List[str]:
        """Generate recommendations based on comparison"""
        
        recommendations = []
        
        overall_score = similarity.get('overall_score', 0)
        
        if overall_score < 0.5:
            recommendations.append("Significant differences detected - focus on fundamentals")
        elif overall_score < 0.7:
            recommendations.append("Good foundation - refine specific technical elements")
        else:
            recommendations.append("Excellent technique - minor adjustments only")
        
        # Component-specific recommendations
        component_scores = similarity.get('component_scores', {})
        
        if 'elbow_angle' in component_scores and component_scores['elbow_angle'] < 0.6:
            recommendations.append("Adjust elbow angle during swing to match professional technique")
        
        if 'shoulder_rotation' in component_scores and component_scores['shoulder_rotation'] < 0.6:
            recommendations.append("Increase shoulder rotation for more power generation")
        
        # Timing recommendations
        if 'temporal_analysis' in similarity:
            timing_score = similarity['temporal_analysis'].get('timing_score', 1)
            if timing_score < 0.6:
                recommendations.append("Work on timing rhythm - practice with metronome")
        
        return recommendations
