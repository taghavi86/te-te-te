"""
Prompt Management - System prompts and prompt templates for AI Coach
"""

from typing import Dict, Any, Optional


class PromptManager:
    """Manages prompts for LLM interactions."""
    
    SYSTEM_PROMPT = """You are an elite table tennis coach and biomechanics analyst. Your role is to provide evidence-based training advice based solely on the video analysis data provided.

RULES:
1. Use ONLY the data from the Analysis Context - do not invent numbers or facts
2. Clearly distinguish between measurements (observed data) and inferences (your analysis)
3. Identify root causes, not just symptoms
4. Prioritize the most important correction first
5. Provide specific, measurable training recommendations
6. Reference specific frames, strokes, or phases when discussing evidence
7. Be encouraging but honest about areas needing improvement
8. Acknowledge strengths as well as weaknesses

YOUR ROLE:
- Elite Table Tennis Coach
- Biomechanics Analyst  
- Evidence-based Training Advisor

RESPONSE FORMAT:
When generating reports, use this JSON structure:
{
  "summary": "Brief overview of the player's technique",
  "primary_issue": "Most important issue to address",
  "root_cause": "Underlying cause of the primary issue",
  "evidence": ["list of specific evidence from video"],
  "secondary_issues": ["other issues identified"],
  "strengths": ["what the player does well"],
  "corrections": ["specific corrections with drills"],
  "training_plan": ["structured training recommendations"],
  "next_session_goal": "specific goal for next practice"
}

Remember: Quality over quantity. One well-executed correction is better than ten vague suggestions."""

    def __init__(self):
        self.system_prompt = self.SYSTEM_PROMPT
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for coaching."""
        return self.system_prompt
    
    def build_report_prompt(self, analysis_context: Dict[str, Any]) -> str:
        """
        Build prompt for generating coach report.
        
        Args:
            analysis_context: Complete analysis data
            
        Returns:
            Formatted prompt string
        """
        prompt = """Analyze the following table tennis stroke analysis data and generate a comprehensive coach report.

ANALYSIS CONTEXT:
"""
        
        # Add video metadata
        if 'video_metadata' in analysis_context:
            meta = analysis_context['video_metadata']
            prompt += f"\nVideo Information:\n"
            prompt += f"- User Video: {meta.get('user_video', 'N/A')}\n"
            prompt += f"- Reference Video: {meta.get('reference_video', 'N/A')}\n"
            prompt += f"- Duration: {meta.get('duration', 'N/A')} seconds\n"
            prompt += f"- FPS: {meta.get('fps', 'N/A')}\n"
        
        # Add stroke data
        if 'stroke_data' in analysis_context:
            strokes = analysis_context['stroke_data']
            prompt += f"\nStroke Analysis:\n"
            prompt += f"- Total Strokes Detected: {len(strokes)}\n"
            
            if strokes:
                prompt += "\nKey Stroke Details:\n"
                for i, stroke in enumerate(strokes[:5]):  # Limit to first 5
                    prompt += f"  Stroke {i+1}: {stroke.get('type', 'Unknown')} at frame {stroke.get('start_frame', 'N/A')}\n"
        
        # Add biomechanical features
        if 'biomechanical_features' in analysis_context:
            features = analysis_context['biomechanical_features']
            prompt += f"\nBiomechanical Measurements:\n"
            
            if 'angles' in features:
                angles = features['angles']
                prompt += f"  Joint Angles:\n"
                for joint, value in list(angles.items())[:5]:
                    prompt += f"    - {joint}: {value:.1f}°\n"
            
            if 'velocities' in features:
                velocities = features['velocities']
                prompt += f"  Velocities:\n"
                for key, value in list(velocities.items())[:3]:
                    prompt += f"    - {key}: {value:.2f}\n"
        
        # Add comparison data
        if 'comparison' in analysis_context:
            comparison = analysis_context['comparison']
            prompt += f"\nComparison with Reference:\n"
            
            if 'similarities' in comparison:
                sim = comparison['similarities']
                prompt += f"  Overall Similarity: {sim.get('overall', 0):.1%}\n"
                
            if 'differences' in comparison:
                diffs = comparison['differences']
                prompt += f"  Key Differences:\n"
                for diff in list(diffs)[:5]:
                    prompt += f"    - {diff.get('feature', 'Unknown')}: {diff.get('difference', 0):.1f}\n"
        
        # Add diagnosis
        if 'diagnosis' in analysis_context:
            diagnosis = analysis_context['diagnosis']
            prompt += f"\nAutomated Diagnosis:\n"
            
            if 'issues' in diagnosis:
                prompt += f"  Issues Identified: {len(diagnosis['issues'])}\n"
                for issue in diagnosis['issues'][:3]:
                    prompt += f"    - {issue.get('type', 'Unknown')} (confidence: {issue.get('confidence', 0):.2f})\n"
            
            if 'root_cause' in diagnosis:
                prompt += f"  Root Cause: {diagnosis['root_cause']}\n"
        
        prompt += """
Based on this analysis, generate a structured coach report in JSON format with:
1. A concise summary
2. The primary issue that needs attention
3. The root cause (not just symptoms)
4. Specific evidence from the video
5. Secondary issues
6. Player's strengths
7. Actionable corrections with drills
8. A structured training plan
9. A specific goal for the next session

Focus on quality over quantity. One well-explained correction is better than ten vague suggestions."""
        
        return prompt
    
    def build_chat_prompt(self, question: str, context: Optional[str] = None) -> str:
        """
        Build prompt for chat interaction.
        
        Args:
            question: User's question
            context: Relevant analysis context
            
        Returns:
            Formatted prompt string
        """
        if context:
            return f"""Context from analysis:
{context}

User Question: {question}

Provide a helpful, evidence-based answer using only the context above. If the context doesn't contain relevant information, say so."""
        else:
            return f"""User Question: {question}

Provide a helpful response based on your expertise as a table tennis coach."""
    
    def build_context_retrieval_prompt(self, question: str) -> str:
        """
        Build prompt for retrieving relevant context.
        
        Args:
            question: User's question
            
        Returns:
            Intent classification
        """
        question_lower = question.lower()
        
        # Simple keyword-based intent detection
        intents = {
            'timing': ['timing', 'when', 'late', 'early', 'tempo', 'rhythm'],
            'angle': ['angle', 'elbow', 'wrist', 'shoulder', 'knee', 'hip'],
            'velocity': ['speed', 'fast', 'slow', 'velocity', 'acceleration'],
            'backswing': ['backswing', 'back swing', 'preparation'],
            'follow_through': ['follow through', 'follow-through', 'finish'],
            'stance': ['stance', 'feet', 'position', 'posture'],
            'rotation': ['rotation', 'twist', 'turn', 'hip rotation', 'shoulder rotation']
        }
        
        detected_intent = []
        for intent, keywords in intents.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_intent.append(intent)
        
        return detected_intent if detected_intent else ['general']
