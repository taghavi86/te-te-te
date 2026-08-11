"""
Advanced Technique Analysis Engine
Evaluates tennis technique using biomechanical principles
Provides detailed feedback and scoring
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class TechniqueScore:
    """Data class for technique scoring"""
    overall_score: float
    component_scores: Dict[str, float]
    feedback: List[str]
    recommendations: List[str]

class TechniqueAnalyzer:
    """Advanced tennis technique analysis system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Biomechanical thresholds for professional technique
        self.optimal_angles = {
            'knee_bend': (120, 150),  # degrees
            'elbow_extension': (160, 180),
            'shoulder_rotation': (90, 120),
            'hip_shoulder_separation': (30, 60),
            'wrist_lag': (20, 45)
        }
        
        # Weight factors for different components
        self.component_weights = {
            'stance': 0.15,
            'backswing': 0.20,
            'contact_point': 0.25,
            'follow_through': 0.20,
            'balance': 0.20
        }
        
        # Professional reference data (would be loaded from database)
        self.pro_references = self._load_pro_references()
    
    def _load_pro_references(self) -> Dict:
        """Load professional player reference data"""
        # In production, this would load from a database
        return {
            'forehand': {
                'avg_knee_angle': 135,
                'avg_elbow_angle': 170,
                'avg_shoulder_rotation': 105,
                'contact_height': 0.8,  # normalized
                'swing_speed': 120  # km/h equivalent
            },
            'backhand': {
                'avg_knee_angle': 140,
                'avg_elbow_angle': 165,
                'avg_shoulder_rotation': 95,
                'contact_height': 0.75,
                'swing_speed': 100
            },
            'serve': {
                'avg_knee_angle': 145,
                'avg_elbow_angle': 175,
                'avg_shoulder_rotation': 135,
                'contact_height': 1.0,
                'swing_speed': 180
            }
        }
    
    def analyze(self, pose_data: Dict, frame_number: int) -> Dict:
        """Comprehensive technique analysis"""
        
        if not pose_data.get('detected', False):
            return {
                'score': 0.0,
                'feedback': ['No pose detected'],
                'recommendations': [],
                'phase': 'unknown'
            }
        
        angles = pose_data.get('angles', [])
        metrics = pose_data.get('tennis_metrics', {})
        
        # Analyze different phases
        phase_analysis = self._identify_phase(metrics, frame_number)
        
        # Component scoring
        component_scores = {
            'stance': self._analyze_stance(pose_data),
            'backswing': self._analyze_backswing(pose_data, phase_analysis),
            'contact_point': self._analyze_contact(pose_data, phase_analysis),
            'follow_through': self._analyze_follow_through(pose_data, phase_analysis),
            'balance': self._analyze_balance(metrics)
        }
        
        # Calculate overall score
        overall_score = sum(
            score * self.component_weights[component]
            for component, score in component_scores.items()
        )
        
        # Generate feedback
        feedback = self._generate_feedback(component_scores, phase_analysis)
        recommendations = self._generate_recommendations(component_scores, angles)
        
        # Compare with professional references
        pro_comparison = self._compare_with_pro(pose_data, phase_analysis)
        
        return {
            'score': round(overall_score * 100, 2),  # Convert to percentage
            'component_scores': {k: round(v * 100, 2) for k, v in component_scores.items()},
            'feedback': feedback,
            'recommendations': recommendations,
            'phase': phase_analysis['current_phase'],
            'pro_comparison': pro_comparison,
            'biomechanics': self._analyze_biomechanics(angles, metrics)
        }
    
    def _identify_phase(self, metrics: Dict, frame_number: int) -> Dict:
        """Identify current stroke phase"""
        
        swing_phase = metrics.get('swing_phase', 'unknown')
        
        phase_details = {
            'current_phase': swing_phase,
            'phase_quality': 0.0,
            'timing': 0.0,
            'transitions': []
        }
        
        # Evaluate phase quality
        if swing_phase == 'ready_position':
            phase_details['phase_quality'] = 1.0 if metrics.get('ready_position', False) else 0.5
        elif swing_phase == 'backswing':
            # Quality based on body rotation
            rotation = metrics.get('body_rotation', 0)
            phase_details['phase_quality'] = min(1.0, rotation / 0.5)
        elif swing_phase == 'contact':
            # Quality based on balance and alignment
            balance = metrics.get('balance_score', 0)
            phase_details['phase_quality'] = balance
        elif swing_phase == 'follow_through':
            phase_details['phase_quality'] = 0.8  # Simplified
        
        return phase_details
    
    def _analyze_stance(self, pose_data: Dict) -> float:
        """Analyze stance quality"""
        
        keypoints = pose_data.get('keypoints', [])
        if len(keypoints) < 24:
            return 0.5
        
        # Check foot positioning and weight distribution
        left_hip = keypoints[23]
        right_hip = keypoints[24]
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        
        # Stance width (distance between hips)
        stance_width = abs(left_hip['x'] - right_hip['x'])
        
        # Shoulder-hip alignment
        shoulder_angle = abs(left_shoulder['y'] - right_shoulder['y'])
        hip_angle = abs(left_hip['y'] - right_hip['y'])
        
        # Score based on athletic stance characteristics
        score = 0.0
        
        # Good stance width
        if 0.15 < stance_width < 0.35:
            score += 0.4
        
        # Proper alignment
        alignment_diff = abs(shoulder_angle - hip_angle)
        if alignment_diff < 0.05:
            score += 0.3
        
        # Weight on balls of feet (estimated from knee bend)
        if len(pose_data.get('angles', [])) > 1:
            knee_angle = pose_data['angles'][1]
            if 120 <= knee_angle <= 150:
                score += 0.3
        
        return min(1.0, score)
    
    def _analyze_backswing(self, pose_data: Dict, phase_info: Dict) -> float:
        """Analyze backswing quality"""
        
        angles = pose_data.get('angles', [])
        if not angles:
            return 0.5
        
        score = 0.0
        
        # Check elbow angle for proper takeback
        if len(angles) > 0:
            elbow_angle = angles[0]
            optimal_range = self.optimal_angles['elbow_extension']
            
            if optimal_range[0] <= elbow_angle <= optimal_range[1]:
                score += 0.4
            elif abs(elbow_angle - np.mean(optimal_range)) < 20:
                score += 0.2
        
        # Check shoulder rotation
        metrics = pose_data.get('tennis_metrics', {})
        body_rotation = metrics.get('body_rotation', 0)
        
        if body_rotation > 0.4:
            score += 0.3
        elif body_rotation > 0.2:
            score += 0.15
        
        # Smooth transition check
        if phase_info.get('phase_quality', 0) > 0.7:
            score += 0.3
        
        return min(1.0, score)
    
    def _analyze_contact(self, pose_data: Dict, phase_info: Dict) -> float:
        """Analyze contact point quality"""
        
        metrics = pose_data.get('tennis_metrics', {})
        angles = pose_data.get('angles', [])
        
        score = 0.0
        
        # Balance at contact is crucial
        balance_score = metrics.get('balance_score', 0)
        score += balance_score * 0.4
        
        # Body rotation at contact
        body_rotation = metrics.get('body_rotation', 0)
        if 0.3 <= body_rotation <= 0.6:
            score += 0.3
        elif 0.2 <= body_rotation <= 0.7:
            score += 0.15
        
        # Arm extension
        if len(angles) > 0:
            arm_angle = angles[0]
            if 160 <= arm_angle <= 180:
                score += 0.3
            elif 150 <= arm_angle <= 170:
                score += 0.15
        
        return min(1.0, score)
    
    def _analyze_follow_through(self, pose_data: Dict, phase_info: Dict) -> float:
        """Analyze follow-through quality"""
        
        metrics = pose_data.get('tennis_metrics', {})
        
        score = 0.0
        
        # Check if follow-through was completed
        if metrics.get('swing_phase') == 'follow_through':
            score += 0.5
            
            # Quality of follow-through
            body_rotation = metrics.get('body_rotation', 0)
            if body_rotation > 0.5:
                score += 0.3
            elif body_rotation > 0.3:
                score += 0.15
            
            # Balance maintenance
            balance = metrics.get('balance_score', 0)
            score += balance * 0.2
        
        return min(1.0, score)
    
    def _analyze_balance(self, metrics: Dict) -> float:
        """Analyze overall balance throughout movement"""
        
        balance_score = metrics.get('balance_score', 0)
        ready_position = metrics.get('ready_position', False)
        
        score = balance_score * 0.7
        
        if ready_position:
            score += 0.3
        
        return min(1.0, score)
    
    def _generate_feedback(self, component_scores: Dict, phase_info: Dict) -> List[str]:
        """Generate specific feedback based on analysis"""
        
        feedback = []
        
        # Overall phase feedback
        phase = phase_info.get('current_phase', 'unknown')
        phase_quality = phase_info.get('phase_quality', 0)
        
        if phase_quality < 0.6:
            feedback.append(f"Improve {phase} execution")
        
        # Component-specific feedback
        if component_scores['stance'] < 0.6:
            feedback.append("Stance needs improvement - widen your base")
        
        if component_scores['backswing'] < 0.6:
            feedback.append("Backswing too short - rotate shoulders more")
        
        if component_scores['contact_point'] < 0.6:
            feedback.append("Contact point inconsistent - focus on timing")
        
        if component_scores['follow_through'] < 0.6:
            feedback.append("Incomplete follow-through - extend through the ball")
        
        if component_scores['balance'] < 0.6:
            feedback.append("Balance issues - keep knees bent and core engaged")
        
        return feedback
    
    def _generate_recommendations(self, component_scores: Dict, 
                                 angles: List[float]) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Prioritize weakest components
        sorted_components = sorted(
            component_scores.items(),
            key=lambda x: x[1]
        )
        
        weakest_component = sorted_components[0][0]
        weakest_score = sorted_components[0][1]
        
        if weakest_score < 0.5:
            if weakest_component == 'stance':
                recommendations.append("Drill: Practice shadow swings focusing on wide stance")
            elif weakest_component == 'backswing':
                recommendations.append("Drill: Use mirror work to check shoulder rotation")
            elif weakest_component == 'contact_point':
                recommendations.append("Drill: Ball toss practice for consistent contact height")
            elif weakest_component == 'follow_through':
                recommendations.append("Drill: Exaggerated follow-through exercises")
            elif weakest_component == 'balance':
                recommendations.append("Drill: Single-leg balance exercises")
        
        # Angle-specific recommendations
        if len(angles) > 0:
            elbow_angle = angles[0]
            if elbow_angle < 150:
                recommendations.append("Keep elbow more extended during swing")
            elif elbow_angle > 175:
                recommendations.append("Maintain slight elbow flexion for control")
        
        return recommendations
    
    def _compare_with_pro(self, pose_data: Dict, phase_info: Dict) -> Dict:
        """Compare technique with professional references"""
        
        comparison = {
            'similarity_score': 0.0,
            'differences': [],
            'matching_elements': []
        }
        
        # Get current phase type (simplified)
        shot_type = 'forehand'  # Would be determined from context
        
        if shot_type not in self.pro_references:
            return comparison
        
        pro_data = self.pro_references[shot_type]
        angles = pose_data.get('angles', [])
        metrics = pose_data.get('tennis_metrics', {})
        
        similarities = []
        differences = []
        
        # Compare knee angle
        if len(angles) > 1:
            knee_angle = angles[1]
            pro_knee = pro_data['avg_knee_angle']
            
            if abs(knee_angle - pro_knee) < 10:
                similarities.append("Knee bend matches professional level")
            else:
                diff = abs(knee_angle - pro_knee)
                differences.append(f"Knee angle differs by {diff:.1f}° from pros")
        
        # Compare shoulder rotation
        body_rotation = metrics.get('body_rotation', 0)
        pro_rotation = pro_data['avg_shoulder_rotation'] / 180  # Normalize
        
        if abs(body_rotation - pro_rotation) < 0.15:
            similarities.append("Shoulder rotation is professional-level")
        else:
            differences.append("Shoulder rotation needs improvement")
        
        # Calculate overall similarity
        if similarities:
            comparison['similarity_score'] = len(similarities) / (len(similarities) + len(differences) + 1)
        
        comparison['matching_elements'] = similarities
        comparison['differences'] = differences
        
        return comparison
    
    def _analyze_biomechanics(self, angles: List[float], metrics: Dict) -> Dict:
        """Detailed biomechanical analysis"""
        
        biomechanics = {
            'kinetic_chain_efficiency': 0.0,
            'joint_loading': {},
            'energy_transfer': 0.0,
            'injury_risk_factors': []
        }
        
        # Kinetic chain analysis (ground up energy transfer)
        if len(angles) >= 3:
            knee_angle = angles[1]
            hip_angle = angles[2] if len(angles) > 2 else 180
            elbow_angle = angles[0]
            
            # Optimal kinetic chain: legs -> hips -> torso -> arm
            chain_score = 0.0
            
            # Leg drive
            if 120 <= knee_angle <= 150:
                chain_score += 0.3
            
            # Hip rotation
            body_rotation = metrics.get('body_rotation', 0)
            if body_rotation > 0.3:
                chain_score += 0.3
            
            # Arm extension timing
            if 160 <= elbow_angle <= 180:
                chain_score += 0.4
            
            biomechanics['kinetic_chain_efficiency'] = chain_score
        
        # Joint loading estimation
        if len(angles) > 0:
            biomechanics['joint_loading'] = {
                'elbow': abs(angles[0] - 180) / 30,  # Higher deviation = more stress
                'shoulder': metrics.get('body_rotation', 0) * 2,
                'knee': abs((angles[1] if len(angles) > 1 else 140) - 135) / 20
            }
        
        # Injury risk assessment
        risk_factors = []
        
        if len(angles) > 0 and angles[0] < 140:
            risk_factors.append("High elbow stress - risk of tennis elbow")
        
        if metrics.get('balance_score', 1) < 0.5:
            risk_factors.append("Poor balance increases ankle injury risk")
        
        biomechanics['injury_risk_factors'] = risk_factors
        
        # Energy transfer efficiency
        if biomechanics['kinetic_chain_efficiency'] > 0.7:
            biomechanics['energy_transfer'] = 0.8
        elif biomechanics['kinetic_chain_efficiency'] > 0.4:
            biomechanics['energy_transfer'] = 0.5
        else:
            biomechanics['energy_transfer'] = 0.3
        
        return biomechanics
