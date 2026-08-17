"""
LLM Client - LM Studio API Integration
Handles communication with local Qwen3.5-9B model
"""

import json
from typing import Optional, Dict, Any, Generator, List
from pathlib import Path

import httpx

from app.core.config import LLMConfig


class LLMClient:
    """Client for LM Studio local API."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.endpoint = config.endpoint
        self.model = config.model
        self.timeout = config.timeout
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        
        self.client = httpx.Client(
            base_url=self.endpoint,
            timeout=httpx.Timeout(self.timeout)
        )
        
        self.conversation_history: List[Dict[str, str]] = []
    
    def chat(self, message: str, context: Optional[str] = None) -> str:
        """
        Send a chat message to the LLM.
        
        Args:
            message: User message
            context: Optional context from analysis
            
        Returns:
            LLM response text
        """
        # Build messages
        messages = self._build_messages(message, context)
        
        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            data = response.json()
            assistant_message = data["choices"][0]["message"]["content"]
            
            # Store in history
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
            
        except httpx.HTTPError as e:
            return f"Error: Failed to connect to LLM. {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def chat_stream(self, message: str, context: Optional[str] = None) -> Generator[str, None, None]:
        """
        Send a chat message and stream the response.
        
        Args:
            message: User message
            context: Optional context
            
        Yields:
            Chunks of LLM response
        """
        messages = self._build_messages(message, context)
        
        try:
            with self.client.stream(
                "POST",
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True
                }
            ) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.HTTPError as e:
            yield f"Error: Failed to connect to LLM. {str(e)}"
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def generate_report(self, analysis_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured coach report from analysis context.
        
        Args:
            analysis_context: Complete analysis data
            
        Returns:
            Structured report dictionary
        """
        from app.llm.prompts import PromptManager
        
        prompt_manager = PromptManager()
        system_prompt = prompt_manager.get_system_prompt()
        user_prompt = prompt_manager.build_report_prompt(analysis_context)
        
        try:
            response = self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.1,  # Lower temperature for structured output
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"}
                }
            )
            response.raise_for_status()
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            
            # Parse JSON response
            report = json.loads(content)
            
            return report
            
        except httpx.HTTPError as e:
            return {"error": f"Failed to connect to LLM: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _build_messages(self, message: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """Build message list for API call."""
        from app.llm.prompts import PromptManager
        
        prompt_manager = PromptManager()
        system_prompt = prompt_manager.get_system_prompt()
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Analysis Context:\n{context}"
            })
        
        # Add conversation history (last N messages)
        max_history = self.config.context.max_messages if hasattr(self.config, 'context') else 10
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
        
        messages.extend(self.conversation_history)
        
        return messages
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def is_available(self) -> bool:
        """Check if LLM server is available."""
        try:
            response = self.client.get("/models")
            return response.status_code == 200
        except:
            return False
