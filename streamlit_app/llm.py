"""Secrets and streaming chat completions from Groq or Gemini, shared by the chat and domain RAG pages."""
import json
import os

import httpx
import streamlit as st

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def get_secret(name, default=None):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:
        pass
    return os.getenv(name, default)


def stream_gemini(api_key, model, messages, system_prompt, temperature):
    """Yield the reply text chunk by chunk as Gemini streams it."""
    payload = {
        "contents": [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": temperature},
    }
    with httpx.stream(
        "POST",
        GEMINI_URL.format(model=model),
        params={"alt": "sse"},
        headers={"x-goog-api-key": api_key},
        json=payload,
        timeout=120,
    ) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(f"Gemini API error {response.status_code}: {response.text[:500]}")
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            chunk = json.loads(line[len("data: "):])
            for candidate in chunk.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if part.get("text") and not part.get("thought"):
                        yield part["text"]


def stream_groq(api_key, model, messages, system_prompt, temperature):
    """Yield the reply text chunk by chunk from Groq's OpenAI-compatible API."""
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": temperature,
        "stream": True,
    }
    with httpx.stream(
        "POST",
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=120,
    ) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:500]}")
        for line in response.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            for choice in chunk.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    yield text
