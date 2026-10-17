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


# Mirrors the look of frontend/index.html: purple gradient, frosted-glass panels,
# gradient user bubbles on the right and glass assistant bubbles on the left.
CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    background-attachment: fixed;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent; }

.app-header {
    background: rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
}
.app-header h1 { font-size: 2.5rem; font-weight: 700; margin: 0; padding: 0; color: #fff; }

.chat-header {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.chat-title { font-size: 1.3rem; font-weight: 600; }
.model-pill {
    background: rgba(76, 175, 80, 0.2);
    border: 1px solid rgba(76, 175, 80, 0.5);
    border-radius: 20px;
    padding: 6px 14px;
    color: #7ee787;
    font-size: 0.85rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(15px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}
[data-testid="stSidebar"] h3 { color: #fff; font-size: 1.15rem; }

/* Organization picker: radio options drawn as the original's tenant cards */
[data-testid="stSidebar"] [role="radiogroup"] { gap: 10px; }
[data-testid="stSidebar"] [role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 14px;
    margin: 0;
    width: 100%;
    transition: all 0.3s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.4);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: rgba(255, 255, 255, 0.3);
    border-color: #ffffff;
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.2);
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] p { font-weight: 700; color: #fff; }
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stCaptionContainer"] p {
    font-weight: 400; color: rgba(255, 255, 255, 0.8); font-size: 0.85rem;
}

/* Sidebar buttons, like the original's .button */
[data-testid="stSidebar"] .stButton button {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    color: #fff;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.4);
    color: #fff;
}

/* Chat bubbles */
[data-testid="stChatMessage"] { background: transparent; padding: 0; margin: 12px 0; width: 100%; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"] { display: none; }
[data-testid="stChatMessageContent"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px 20px 20px 5px;
    padding: 15px 20px;
    max-width: 80%;
    flex: 0 1 auto;
    margin: 0 auto 0 0 !important;
    color: #fff;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
    border-radius: 20px 20px 5px 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    margin: 0 0 0 auto !important;
}
[data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li { color: #fff; }
[data-testid="stChatMessageContent"] code { background: rgba(0, 0, 0, 0.25); color: #fff; }

/* Message input */
[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 25px;
    backdrop-filter: blur(10px);
}
[data-testid="stChatInput"]:focus-within { border-color: rgba(255, 255, 255, 0.5); }
[data-testid="stChatInput"] textarea { color: #fff; }
[data-testid="stChatInput"] textarea::placeholder { color: rgba(255, 255, 255, 0.6); }
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 50%;
    color: #fff;
}
</style>
"""

WELCOME = (
    "👋 Welcome to the Multi-Tenant Chatbot!\n\n"
    "**Features:**  \n"
    "🏢 Multi-tenant system prompts  \n"
    "⚡ Fast hosted LLM  \n"
    "🧠 Conversation memory\n\n"
    "Pick an organization in the sidebar and ask me anything."
)

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="app-header"><h1>🤖 AI Chatbot</h1></div>', unsafe_allow_html=True)

if get_secret("GROQ_API_KEY"):
    api_key = get_secret("GROQ_API_KEY")
    model = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
    provider = "Groq"
    stream_reply = stream_groq
else:
    api_key = get_secret("GEMINI_API_KEY")
    model = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
    provider = "Gemini"
    stream_reply = stream_gemini
tenants = load_tenants()

with st.sidebar:
    st.markdown("### 🏢 Organizations")
    tenant_id = st.radio(
        "Organization",
        options=list(tenants),
        format_func=lambda t: tenants[t]["tenant_name"],
        captions=[
            f"{tenants[t].get('description', '')}  \n"
            f"🌡️ temp {tenants[t].get('llm', {}).get('temperature', 0.7)} · ⚡ {provider}"
            for t in tenants
        ],
        label_visibility="collapsed",
    )
    tenant = tenants[tenant_id]
    st.write("")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown(
    f'<div class="chat-header"><div class="chat-title">💬 {tenant["tenant_name"]}</div>'
    f'<div class="model-pill">⚡ {provider} · {model}</div></div>',
    unsafe_allow_html=True,
)

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

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)

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
