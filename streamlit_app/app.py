"""
Streamlit version of the chatbot, for deploying to Streamlit Community Cloud.

The local app talks to Ollama, which a cloud host can't reach, so this version
calls a hosted API instead: Groq when GROQ_API_KEY is set, otherwise Gemini.
Tenant system prompts and temperatures are read from the same
config/tenants/*.yaml files the FastAPI backend uses.

Run locally:  streamlit run streamlit_app/app.py
"""
import json
import os
from pathlib import Path

import httpx
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
TENANTS_DIR = ROOT / "config" / "tenants"
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


@st.cache_data
def load_tenants():
    tenants = {}
    for path in sorted(TENANTS_DIR.glob("*.yaml")):
        with open(path) as f:
            config = yaml.safe_load(f)
        tenants[config["tenant_id"]] = config
    return tenants


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


st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot")

if get_secret("GROQ_API_KEY"):
    api_key = get_secret("GROQ_API_KEY")
    model = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
    stream_reply = stream_groq
else:
    api_key = get_secret("GEMINI_API_KEY")
    model = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
    stream_reply = stream_gemini
tenants = load_tenants()

with st.sidebar:
    tenant_id = st.selectbox(
        "Organization",
        options=list(tenants),
        format_func=lambda t: tenants[t]["tenant_name"],
    )
    tenant = tenants[tenant_id]
    st.caption(tenant.get("description", ""))
    st.caption(f"Model: `{model}`")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.error(
        "No API key found. Set `GROQ_API_KEY` (https://console.groq.com/keys) or `GEMINI_API_KEY` "
        "(https://aistudio.google.com/apikey) in `.streamlit/secrets.toml` (local) or the app's "
        "Secrets settings (Streamlit Cloud)."
    )
    st.stop()

# Chat history lives in the browser session; switching tenant starts a new chat.
if st.session_state.get("tenant_id") != tenant_id:
    st.session_state.tenant_id = tenant_id
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                stream_reply(
                    api_key,
                    model,
                    st.session_state.messages,
                    tenant["system_prompt"],
                    tenant.get("llm", {}).get("temperature", 0.7),
                )
            )
        except Exception as e:
            st.session_state.messages.pop()  # let the user retry the same message
            st.error(str(e))
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": reply})
