import logging
from typing import Optional, List

from models.chat import ChatMessage, MessageRole
from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful AI assistant. Provide clear, useful, and engaging responses."

BRIEF_PROMPT = """You are a helpful AI assistant. Keep responses concise and direct:
- Maximum 2-3 sentences
- Get straight to the point
- Use simple, clear language
- No unnecessary explanations"""

# How many earlier messages from the session to send as context
HISTORY_LIMIT = 20


class OllamaAgent:
    """Agent using a local Ollama model, with conversation history from the database."""

    def __init__(self, model: Optional[str] = None):
        self.ollama_service = OllamaService(model=model)

    async def run(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None,
        brief_mode: bool = False,
    ) -> str:
        """Answer `message`, using earlier user/assistant turns in `history` as context."""
        try:
            messages = [{"role": "system", "content": BRIEF_PROMPT if brief_mode else SYSTEM_PROMPT}]
            for past in (history or [])[-HISTORY_LIMIT:]:
                if past.role in (MessageRole.USER, MessageRole.ASSISTANT):
                    messages.append({"role": past.role.value, "content": past.content})
            messages.append({"role": "user", "content": message})

            return await self.ollama_service.chat(
                messages,
                max_tokens=200 if brief_mode else 1000,
                temperature=0.2 if brief_mode else 0.3,
            )

        except Exception as e:
            logger.error(f"Error in Ollama agent: {e}")
            return f"I couldn't get a response from the local model. {e}"

    async def test_connection(self) -> bool:
        """Test that Ollama is reachable and the model is pulled"""
        return await self.ollama_service.test_connection()
