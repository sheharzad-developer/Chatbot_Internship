import logging
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

logger = logging.getLogger(__name__)

class SimpleAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.google_ai_generative,
            temperature=0.7
        )
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run a simple direct response without external service calls"""
        logger.info(f"Processing message: {message}")
        
        try:
            system_prompt = """You are a helpful AI assistant. Provide clear, helpful, and conversational responses to user questions. 
            Be friendly and informative while keeping responses concise and relevant."""
            
            full_prompt = f"{system_prompt}\n\nUser: {message}\n\nAssistant:"
            
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            logger.info("Response generated successfully")
            return response.content
            
        except Exception as e:
            logger.error(f"Error in simple agent: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again." 