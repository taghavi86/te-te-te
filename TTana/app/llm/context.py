"""
Context Builder - Builds relevant context for LLM from analysis data
"""

from typing import Dict, Any, List, Optional


class ContextBuilder:
    """Builds contextual information for LLM queries."""
    
    def __init__(self):
        self.analysis_context: Optional[Dict[str, Any]] = None
    
    def set_analysis_context(self, context: Dict[str, Any]):
        """Set the complete analysis context."""
        self.analysis_context = context
    
    def build_full_context(self) -> str:
        """Build complete context string for LLM."""
        if not self.analysis_context:
            return "No analysis data available."
        
        context_parts = []
        
        # Video metadata
        if 'video_metadata' in self.analysis_context:
            meta = self.analysis_context['video_metadata']
            context_parts.append(
                f"Video: {meta.get('user_video', 'N/A')} | "
                f"Reference: {meta.get('reference_video', 'N/A')} | "
                f"Duration: {meta.get('duration', 0):.1f}s | "
                f"FPS: {meta.get('fps', 30)}"
            )
        
        # Stroke summary
        if 'stroke_data' in self.analysis_context:
            strokes = self.analysis_context['stroke_data']
            context_parts.append(f"Strokes detected: {len(strokes)}")
        
        # Key biomechanical features
        if 'biomechanical_features' in self.analysis_context:
            features = self.analysis_context['biomechanical_features']
            
            if 'angles' in features:
                angles = features['angles']
                angle_str = ", ".join([
                    f"{k}: {v:.1f}°" for k, v in list(angles.items())[:5]
                ])
                context_parts.append(f"Angles: {angle_str}")
            
            if 'velocities' in features:
                velocities = features['velocities']
                vel_str = ", ".join([
                    f"{k}: {v:.2f}" for k, v in list(velocities.items())[:3]
                ])
                context_parts.append(f"Velocities: {vel_str}")
        
        # Comparison results
        if 'comparison' in self.analysis_context:
            comp = self.analysis_context['comparison']
            if 'overall_similarity' in comp:
                context_parts.append(
                    f"Similarity to pro: {comp['overall_similarity']:.1%}"
                )
        
        # Diagnosis
        if 'diagnosis' in self.analysis_context:
            diag = self.analysis_context['diagnosis']
            if 'primary_issue' in diag:
                context_parts.append(f"Primary issue: {diag['primary_issue']}")
            if 'root_cause' in diag:
                context_parts.append(f"Root cause: {diag['root_cause']}")
        
        return " | ".join(context_parts)
    
    def get_relevant_context(self, question: str) -> str:
        """
        Get context relevant to a specific question.
        
        Args:
            question: User's question
            
        Returns:
            Relevant context string
        """
        from app.llm.prompts import PromptManager
        
        prompt_manager = PromptManager()
        intents = prompt_manager.build_context_retrieval_prompt(question)
        
        relevant_data = []
        
        if not self.analysis_context:
            return "No analysis data available."
        
        # Add data based on detected intents
        if 'timing' in intents or 'general' in intents:
            if 'phase_data' in self.analysis_context:
                relevant_data.append(
                    f"Phase timing: {self.analysis_context['phase_data']}"
                )
            if 'dtw_alignment' in self.analysis_context:
                relevant_data.append(
                    f"DTW alignment: {self.analysis_context['dtw_alignment']}"
                )
        
        if 'angle' in intents or 'general' in intents:
            if 'biomechanical_features' in self.analysis_context:
                features = self.analysis_context['biomechanical_features']
                if 'angles' in features:
                    relevant_data.append(f"Joint angles: {features['angles']}")
        
        if 'velocity' in intents or 'general' in intents:
            if 'biomechanical_features' in self.analysis_context:
                features = self.analysis_context['biomechanical_features']
                if 'velocities' in features:
                    relevant_data.append(f"Velocities: {features['velocities']}")
        
        if 'backswing' in intents or 'follow_through' in intents:
            if 'stroke_data' in self.analysis_context:
                relevant_data.append(f"Stroke phases: {self.analysis_context['stroke_data']}")
        
        if 'rotation' in intents or 'general' in intents:
            if 'biomechanical_features' in self.analysis_context:
                features = self.analysis_context['biomechanical_features']
                if 'rotations' in features:
                    relevant_data.append(f"Rotations: {features['rotations']}")
        
        # Always include diagnosis
        if 'diagnosis' in self.analysis_context:
            relevant_data.append(f"Diagnosis: {self.analysis_context['diagnosis']}")
        
        return "\n".join(relevant_data) if relevant_data else self.build_full_context()
    
    def get_evidence_for_issue(self, issue_type: str) -> List[Dict[str, Any]]:
        """
        Get video evidence for a specific issue.
        
        Args:
            issue_type: Type of issue to find evidence for
            
        Returns:
            List of evidence dictionaries with frame ranges
        """
        evidence = []
        
        if not self.analysis_context:
            return evidence
        
        # Search stroke data for relevant evidence
        if 'stroke_data' in self.analysis_context:
            for stroke in self.analysis_context['stroke_data']:
                if stroke.get('type') == issue_type:
                    evidence.append({
                        'stroke_id': stroke.get('id'),
                        'start_frame': stroke.get('start_frame'),
                        'end_frame': stroke.get('end_frame'),
                        'phase': stroke.get('phase')
                    })
        
        # Search diagnosis for evidence
        if 'diagnosis' in self.analysis_context:
            diag = self.analysis_context['diagnosis']
            if 'issues' in diag:
                for issue in diag['issues']:
                    if issue.get('type') == issue_type:
                        if 'evidence' in issue:
                            evidence.extend(issue['evidence'])
        
        return evidence[:5]  # Limit to 5 pieces of evidence
