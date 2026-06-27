import os
import logging
from typing import Optional, Dict, Any
from google import genai
from google.genai import types

from config.settings import settings

logger = logging.getLogger(__name__)

class GeminiService:
    """Direct Google Gemini API service using google-genai client"""
    
    def __init__(self):
        # Set the API key as environment variable for the client
        if settings.google_ai_generative:
            os.environ['GEMINI_API_KEY'] = settings.google_ai_generative

        # Lazily constructed so the app can boot without a Google API key.
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not settings.google_ai_generative:
                raise RuntimeError(
                    "GOOGLE_AI_GENERATIVE is not set. Add a Google Gemini API key "
                    "to your .env to enable the Gemini agent."
                )
            self._client = genai.Client()
        return self._client
        
    async def generate_content(
        self, 
        prompt: str, 
        model: str = "gemini-2.5-flash",
        disable_thinking: bool = True
    ) -> str:
        """
        Generate content using Google Gemini API
        
        Args:
            prompt: The input prompt/message
            model: The Gemini model to use (default: gemini-2.5-flash)
            disable_thinking: Whether to disable thinking mode (default: True)
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating content with model: {model}")
            
            if disable_thinking:
                # Use configuration to disable thinking
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
                
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
            else:
                # Basic generation without special config
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt
                )
            
            logger.info("Content generated successfully")
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating content with Gemini: {e}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def generate_simple(self, prompt: str) -> str:
        """Simple content generation without configuration"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Error in simple generation: {e}")
            raise Exception(f"Gemini API error: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test if the Gemini API connection is working"""
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Say 'Hello' in one word"
            )
            return bool(response.text)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False 