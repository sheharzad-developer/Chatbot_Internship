import asyncio
import logging
import aiohttp
import json
from typing import Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

class DeepSeekService:
    """Service for interacting with DeepSeek API"""
    
    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_content(
        self, 
        prompt: str, 
        model: str = "deepseek-chat",
        max_tokens: Optional[int] = 1000,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """Generate content using DeepSeek API"""
        try:
            url = f"{self.base_url}/v1/chat/completions"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        return content.strip()
                    else:
                        error_text = await response.text()
                        logger.error(f"DeepSeek API error {response.status}: {error_text}")
                        raise Exception(f"DeepSeek API error: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"Error generating content with DeepSeek: {e}")
            raise Exception(f"DeepSeek API error: {str(e)}")
    
    async def generate_simple(self, message: str) -> str:
        """Simple generation without complex parameters"""
        return await self.generate_content(
            prompt=f"Please provide a helpful response to: {message}",
            model="deepseek-chat",
            max_tokens=500,
            temperature=0.3
        )
    
    def test_connection(self) -> bool:
        """Test if DeepSeek API connection is working"""
        try:
            # Simple synchronous test
            import requests
            url = f"{self.base_url}/v1/models"
            response = requests.get(url, headers=self.headers, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"DeepSeek connection test failed: {e}")
            return False
    
    async def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            url = f"{self.base_url}/v1/models"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [model["id"] for model in data.get("data", [])]
                    else:
                        logger.error(f"Failed to get models: {response.status}")
                        return ["deepseek-chat"]  # Default fallback
                        
        except Exception as e:
            logger.error(f"Error getting DeepSeek models: {e}")
            return ["deepseek-chat"]  # Default fallback 