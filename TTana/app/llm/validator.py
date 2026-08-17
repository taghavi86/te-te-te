"""
Response Validator - Validates and parses LLM responses
"""

import json
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError


class CoachReport(BaseModel):
    """Structured coach report model."""
    
    summary: str = Field(..., description="Brief overview of the player's technique")
    primary_issue: str = Field(..., description="Most important issue to address")
    root_cause: str = Field(..., description="Underlying cause of the primary issue")
    evidence: list[str] = Field(default_factory=list, description="List of specific evidence from video")
    secondary_issues: list[str] = Field(default_factory=list, description="Other issues identified")
    strengths: list[str] = Field(default_factory=list, description="What the player does well")
    corrections: list[str] = Field(default_factory=list, description="Specific corrections with drills")
    training_plan: list[str] = Field(default_factory=list, description="Structured training recommendations")
    next_session_goal: str = Field(..., description="Specific goal for next practice")


class ResponseValidator:
    """Validates and parses LLM responses."""
    
    @staticmethod
    def validate_report(response: str) -> Tuple[bool, Optional[CoachReport], str]:
        """
        Validate a coach report response.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Tuple of (is_valid, parsed_report, error_message)
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)
            
            # Validate against schema
            report = CoachReport(**data)
            
            return True, report, ""
            
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"
        except ValidationError as e:
            return False, None, f"Schema validation error: {str(e)}"
        except Exception as e:
            return False, None, f"Unexpected error: {str(e)}"
    
    @staticmethod
    def extract_json_from_response(response: str) -> Optional[str]:
        """
        Extract JSON from a response that may contain additional text.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted JSON string or None
        """
        # Try to find JSON block
        start_chars = ['{', '[']
        end_chars = ['}', ']']
        
        start_idx = -1
        end_idx = -1
        
        # Find first opening bracket
        for char in start_chars:
            idx = response.find(char)
            if idx != -1 and (start_idx == -1 or idx < start_idx):
                start_idx = idx
        
        if start_idx == -1:
            return None
        
        # Find matching closing bracket
        bracket_type = response[start_idx]
        expected_close = '}' if bracket_type == '{' else ']'
        depth = 0
        
        for i in range(start_idx, len(response)):
            if response[i] == bracket_type:
                depth += 1
            elif response[i] == expected_close:
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        
        if end_idx == -1:
            return None
        
        return response[start_idx:end_idx]
    
    @staticmethod
    def validate_chat_response(response: str) -> Tuple[bool, str]:
        """
        Validate a chat response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Tuple of (is_valid, cleaned_response)
        """
        if not response or not response.strip():
            return False, ""
        
        # Clean up response
        cleaned = response.strip()
        
        # Remove any markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
                cleaned = "\n".join(lines)
        
        return True, cleaned
    
    @staticmethod
    def check_response_quality(report: CoachReport) -> Dict[str, bool]:
        """
        Check quality aspects of a coach report.
        
        Args:
            report: Parsed coach report
            
        Returns:
            Dictionary of quality checks
        """
        checks = {
            'has_summary': bool(report.summary and len(report.summary) > 20),
            'has_primary_issue': bool(report.primary_issue),
            'has_root_cause': bool(report.root_cause),
            'has_evidence': len(report.evidence) > 0,
            'has_strengths': len(report.strengths) > 0,
            'has_corrections': len(report.corrections) > 0,
            'has_training_plan': len(report.training_plan) > 0,
            'has_goal': bool(report.next_session_goal),
            'evidence_count_adequate': len(report.evidence) >= 2,
            'corrections_count_adequate': len(report.corrections) >= 1,
        }
        
        return checks
    
    @staticmethod
    def format_report_for_display(report: CoachReport) -> str:
        """
        Format a coach report for display in UI.
        
        Args:
            report: Parsed coach report
            
        Returns:
            Formatted string
        """
        lines = [
            "=" * 60,
            "AI COACH REPORT",
            "=" * 60,
            "",
            "📋 SUMMARY",
            "-" * 40,
            report.summary,
            "",
            "🎯 PRIMARY ISSUE",
            "-" * 40,
            report.primary_issue,
            "",
            "🔍 ROOT CAUSE",
            "-" * 40,
            report.root_cause,
            "",
            "📊 EVIDENCE",
            "-" * 40,
        ]
        
        for i, evidence in enumerate(report.evidence, 1):
            lines.append(f"{i}. {evidence}")
        
        lines.extend([
            "",
            "💪 STRENGTHS",
            "-" * 40,
        ])
        
        for strength in report.strengths:
            lines.append(f"✓ {strength}")
        
        lines.extend([
            "",
            "⚠️ SECONDARY ISSUES",
            "-" * 40,
        ])
        
        for issue in report.secondary_issues:
            lines.append(f"• {issue}")
        
        lines.extend([
            "",
            "🔧 CORRECTIONS",
            "-" * 40,
        ])
        
        for i, correction in enumerate(report.corrections, 1):
            lines.append(f"{i}. {correction}")
        
        lines.extend([
            "",
            "📅 TRAINING PLAN",
            "-" * 40,
        ])
        
        for i, plan in enumerate(report.training_plan, 1):
            lines.append(f"{i}. {plan}")
        
        lines.extend([
            "",
            "🎯 NEXT SESSION GOAL",
            "-" * 40,
            report.next_session_goal,
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
