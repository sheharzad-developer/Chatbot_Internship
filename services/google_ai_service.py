import asyncio
import logging
import aiohttp
import json
from typing import Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

class GoogleAIService:
    """Service for interacting with Google AI Studio (Free Gemini API)"""
    
    def __init__(self):
        self.api_key = settings.google_ai_generative  # We'll use the existing Google key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def generate_content(
        self, 
        prompt: str, 
        model: str = "gemini-1.5-flash",
        max_output_tokens: Optional[int] = 1000,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """Generate content using Google AI Studio (Free Gemini API)"""
        try:
            url = f"{self.base_url}/models/{model}:generateContent"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_output_tokens,
                    "temperature": temperature
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            params = {
                "key": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, params=params, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        return content.strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"Google AI API error {response.status}: {error_text}")
                        raise Exception(f"Google AI API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error generating content with Google AI: {e}")
            raise Exception(f"Google AI API error: {str(e)}")
    
    async def generate_simple(self, message: str) -> str:
        """Simple generation without complex parameters"""
        return await self.generate_content(
            prompt=f"Please provide a helpful response to: {message}",
            model="gemini-1.5-flash",
            max_output_tokens=500,
            temperature=0.3
        )
    
    def test_connection(self) -> bool:
        """Test if Google AI API connection is working"""
        try:
            # Simple synchronous test
            import requests
            url = f"{self.base_url}/models"
            response = requests.get(url, params={"key": self.api_key}, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Google AI connection test failed: {e}")
            return False 