import logging
from typing import Optional
from services.groq_service import GroqService

logger = logging.getLogger(__name__)

class GroqAgent:
    """Agent using FREE Groq API - unlimited fast responses!"""
    
    def __init__(self):
        self.groq_service = GroqService()
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run agent with free Groq API"""
        try:
            # Standard prompt for helpful responses
            system_prompt = "You are a helpful AI assistant. Provide clear, useful, and engaging responses."
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response with Groq (Free & Fast!)
            response = await self.groq_service.generate_content(
                prompt=full_prompt,
                model="llama-3.3-70b-versatile",  # High-quality model
                max_tokens=800,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Groq agent: {e}")
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
            
            response = await self.groq_service.generate_content(
                prompt=full_prompt,
                model="llama-3.3-70b-versatile",
                max_tokens=200,
                temperature=0.2
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in Groq brief mode: {e}")
            return "Having connection issues. Please try again!"
    
    async def run_basic(self, message: str) -> str:
        """Run agent with basic Groq generation"""
        try:
            response = await self.groq_service.generate_simple(message)
            return response
        except Exception as e:
            logger.error(f"Error in basic Groq generation: {e}")
            return "Connection error. Please try again!"
    
    def test_connection(self) -> bool:
        """Test if Groq API connection is working"""
        return self.groq_service.test_connection() 