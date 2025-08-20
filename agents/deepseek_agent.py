import logging
from typing import Optional
from services.deepseek_service import DeepSeekService

logger = logging.getLogger(__name__)

class DeepSeekAgent:
    """Agent using DeepSeek API - fast and reliable alternative"""
    
    def __init__(self):
        self.deepseek_service = DeepSeekService()
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run agent with DeepSeek API"""
        try:
            # Optimized prompt for clear, helpful responses
            system_prompt = "You are a helpful AI assistant. Provide clear, concise, and useful responses."
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            # Generate response with DeepSeek
            response = await self.deepseek_service.generate_content(
                prompt=full_prompt,
                model="deepseek-chat",
                max_tokens=800,
                temperature=0.3
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in DeepSeek agent: {e}")
            return "I'm sorry, I encountered an issue while processing your request. Please try again."
    
    async def run_brief(self, message: str) -> str:
        """Run agent with brief responses for clients who prefer shorter answers"""
        try:
            brief_prompt = """You are a helpful AI assistant. Keep responses concise and direct:
- Maximum 2-3 sentences
- Get straight to the point
- Use simple, clear language
- No unnecessary explanations"""
            
            full_prompt = f"{brief_prompt}\n\nUser: {message}\n\nAssistant:"
            
            response = await self.deepseek_service.generate_content(
                prompt=full_prompt,
                model="deepseek-chat",
                max_tokens=200,
                temperature=0.2
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in DeepSeek brief mode: {e}")
            return "Error processing request. Please try again."
    
    async def run_basic(self, message: str) -> str:
        """Run agent with basic DeepSeek generation"""
        try:
            response = await self.deepseek_service.generate_simple(message)
            return response
        except Exception as e:
            logger.error(f"Error in basic DeepSeek generation: {e}")
            return "Error processing request. Please try again."
    
    def test_connection(self) -> bool:
        """Test if DeepSeek API connection is working"""
        return self.deepseek_service.test_connection() 