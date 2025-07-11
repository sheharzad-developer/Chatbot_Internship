import logging
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

logger = logging.getLogger(__name__)

class SimpleAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Faster model
            google_api_key=settings.google_ai_generative,
            temperature=0.3,  # Reduced for faster, more focused responses
            max_tokens=1024,  # Limit token count for speed
        )
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run a simple direct response optimized for speed"""
        try:
            # Shorter, optimized prompt for faster processing
            system_prompt = "You are a helpful AI assistant. Provide clear, concise responses."
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error in simple agent: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again." 