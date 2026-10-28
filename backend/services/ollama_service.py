import logging
from typing import Optional, List, Dict

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaService:
    """Service for a locally running Ollama server (https://ollama.com)."""

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model or settings.ollama_model
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = settings.ollama_timeout

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = 1000,
        temperature: float = 0.3,
    ) -> str:
        """Send a list of {role, content} messages and return the reply text."""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
            if response.status_code != 200:
                raise Exception(f"{response.status_code} - {response.text}")
            return response.json().get("message", {}).get("content", "").strip()
        except httpx.ConnectError:
            raise Exception(
                f"Cannot reach Ollama at {self.base_url}. Is it running? Start it with `ollama serve`."
            )
        except Exception as e:
            logger.error(f"Error generating content with Ollama: {e}")
            raise Exception(f"Ollama error: {str(e)}")

    async def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = 1000,
        temperature: float = 0.3,
        **kwargs
    ) -> str:
        """Single-prompt generation (interface used by ToolAwareAgent)."""
        return await self.chat(
            [{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def list_models(self) -> List[str]:
        """Names of models pulled into the local Ollama server."""
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]

    async def test_connection(self) -> bool:
        """Check that Ollama is up and the configured model is pulled."""
        try:
            models = await self.list_models()
            wanted = self.model if ":" in self.model else f"{self.model}:latest"
            return wanted in models
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
