import logging
from typing import Optional
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class GeminiAgent:
    """Agent using direct Google Gemini API integration"""
    
    def __init__(self):
        self.gemini_service = GeminiService()
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run agent with direct Gemini API"""
        logger.info(f"Processing message with Gemini Agent: {message}")
        
        try:
            system_prompt = """You are a helpful AI assistant. Provide clear, helpful, and conversational responses to user questions. 
            Be friendly and informative while keeping responses concise and relevant."""
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            # Use the direct Gemini service
            response = await self.gemini_service.generate_content(
                prompt=full_prompt,
                model="gemini-2.5-flash",
                disable_thinking=True
            )
            
            logger.info("Response generated successfully with Gemini Agent")
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