import logging
from typing import Optional
from services.google_ai_service import GoogleAIService

logger = logging.getLogger(__name__)

class GoogleAIAgent:
    """Agent using free Google AI Studio - completely free with no limits!"""
    
    def __init__(self):
        self.google_ai_service = GoogleAIService()
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run agent with free Google AI Studio"""
        try:
            # Standard prompt for helpful responses
            system_prompt = "You are a helpful AI assistant. Provide clear, useful, and engaging responses."
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response with Google AI Studio (Free!)
            response = await self.google_ai_service.generate_content(
                prompt=full_prompt,
                model="gemini-1.5-flash",
                max_output_tokens=800,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Google AI agent: {e}")
            return "I'm having trouble connecting right now. Please try again in a moment!"
    
    async def run_brief(self, message: str) -> str:
        """Run agent with brief responses for clients who prefer shorter answers"""
        try:
            brief_prompt = """You are a helpful AI assistant. Keep responses concise and direct:
- Maximum 2-3 sentences
- Get straight to the point
- Use simple, clear language
- No unnecessary explanations"""
            
            full_prompt = f"{brief_prompt}\n\nUser: {message}\n\nAssistant:"
            
            response = await self.google_ai_service.generate_content(
                prompt=full_prompt,
                model="gemini-1.5-flash",
                max_output_tokens=200,
                temperature=0.2
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Google AI brief mode: {e}")
            return "Having connection issues. Please try again!"
    
    async def run_basic(self, message: str) -> str:
        """Run agent with basic Google AI generation"""
        try:
            response = await self.google_ai_service.generate_simple(message)
            return response
        except Exception as e:
            logger.error(f"Error in basic Google AI generation: {e}")
            return "Connection error. Please try again!"
    
    def test_connection(self) -> bool:
        """Test if Google AI API connection is working"""
        return self.google_ai_service.test_connection() 