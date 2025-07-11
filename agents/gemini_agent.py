import logging
from typing import Optional
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class GeminiAgent:
    """Agent using direct Google Gemini API integration - optimized for speed"""
    
    def __init__(self):
        self.gemini_service = GeminiService()
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run agent with direct Gemini API - speed optimized"""
        try:
            # Shorter prompt for faster processing
            system_prompt = "You are a helpful AI assistant. Provide clear, concise responses."
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            # Use faster model and disable thinking for speed
            response = await self.gemini_service.generate_content(
                prompt=full_prompt,
                model="gemini-1.5-flash",  # Use faster model instead of 2.5-flash
                disable_thinking=True
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Gemini agent: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    async def run_basic(self, message: str) -> str:
        """Run agent with basic Gemini generation (no thinking config)"""
        try:
            response = await self.gemini_service.generate_simple(message)
            return response
        except Exception as e:
            logger.error(f"Error in basic Gemini generation: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    def test_connection(self) -> bool:
        """Test if Gemini API connection is working"""
        return self.gemini_service.test_connection() 