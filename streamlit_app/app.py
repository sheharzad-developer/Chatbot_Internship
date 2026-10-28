"""
Streamlit version of the chatbot, for deploying to Streamlit Community Cloud.

The local app talks to Ollama, which a cloud host can't reach, so this version
calls a hosted API instead: Groq when GROQ_API_KEY is set, otherwise Gemini.
Tenant system prompts and temperatures are read from the same
backend/config/tenants/*.yaml files the FastAPI backend uses.

Run locally:  streamlit run streamlit_app/app.py
"""
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
TENANTS_DIR = ROOT / "backend" / "config" / "tenants"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
HISTORY_LIMIT = 20


def get_secret(name, default=None):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except FileNotFoundError:
        pass
    return os.getenv(name, default)


# Same guidance backend/agents/tool_aware_agent.py adds when a tenant has no tools connected.
# This app never connects tools, so without it the model role-plays the tools named in the
# tenant prompt and invents their results.
NO_TOOLS_GUIDANCE = (
    "\n\nIMPORTANT: No tools, internal databases or enterprise systems are connected right now. "
    "Answer from your general knowledge only. Never say you are using a tool, and never invent "
    "search results, internal company figures, policy or compliance checks. If a question needs "
    "internal company data, say you don't have access to it. When prices or costs would help "
    "(hotels, flights, trips, products), do include typical price ranges from general knowledge, "
    "label them as approximate, and suggest checking a live source for current prices."
)
FORMAT_GUIDANCE = (
    "\n\nFormat answers in clean Markdown: a one-line intro, short headings or numbered lists "
    "where they help, and tables only for real comparisons. Keep it concise."
)


def build_system_prompt(tenant):
    return tenant["system_prompt"] + NO_TOOLS_GUIDANCE + FORMAT_GUIDANCE


def escape_dollars(chunks):
    """st.markdown treats text between $ signs as LaTeX, which garbles prices like "$200 – $400"."""
    for chunk in chunks:
        yield chunk.replace("$", "\\$")


# Chat history: same tables as backend/database/sqlite_db.py, but synchronous and without the
# backend's dependencies, so it runs on Streamlit Community Cloud as-is.
HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    user_id     TEXT,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chat_messages (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT NOT NULL UNIQUE,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON chat_sessions(user_id, updated_at);
"""


@contextmanager
def history_db():
    """Open the history file, commit on success, and always close it."""
    path = Path(get_secret("CHAT_HISTORY_DB", str(ROOT / "streamlit_app" / "data" / "chat_history.db")))
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(HISTORY_SCHEMA)
        with conn:
            yield conn
    finally:
        conn.close()


def create_session(user_id, tenant_id, first_message):
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    title = first_message.strip().replace("\n", " ")
    title = title[:45] + "…" if len(title) > 45 else title
    with history_db() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (session_id, tenant_id, user_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, tenant_id, user_id, title, now, now),
        )
    return session_id


def add_message(session_id, role, content):
    now = datetime.utcnow().isoformat()
    with history_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, role, content, now),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (now, session_id))


def list_sessions(user_id):
    with history_db() as conn:
        return conn.execute(
            "SELECT session_id, tenant_id, title, updated_at FROM chat_sessions "
            "WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, HISTORY_LIMIT),
        ).fetchall()


def get_session_tenant(session_id, user_id):
    """Tenant of a saved chat, or None if it doesn't exist or belongs to someone else."""
    with history_db() as conn:
        row = conn.execute(
            "SELECT tenant_id FROM chat_sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        ).fetchone()
    return row["tenant_id"] if row else None


def load_messages(session_id, user_id):
    with history_db() as conn:
        return [
            {"role": m["role"], "content": m["content"]}
            for m in conn.execute(
                "SELECT m.role, m.content FROM chat_messages m "
                "JOIN chat_sessions s ON s.session_id = m.session_id "
                "WHERE m.session_id = ? AND s.user_id = ? ORDER BY m.seq",
                (session_id, user_id),
            )
        ]


def delete_session(session_id, user_id):
    with history_db() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))


def start_new_chat():
    st.session_state.messages = []
    st.session_state.session_id = None
    st.query_params.pop("chat", None)


def open_session(session_id, tenant_id, user_id):
    st.session_state.messages = load_messages(session_id, user_id)
    st.session_state.session_id = session_id
    # Keep the open chat in the URL so a page refresh reopens it
    st.query_params["chat"] = session_id
    # Switch the organization picker to the chat's tenant without triggering a reset
    st.session_state.tenant_id = tenant_id
    st.session_state.tenant_picker = tenant_id


def remove_session(session_id, user_id):
    delete_session(session_id, user_id)
    if st.session_state.get("session_id") == session_id:
        start_new_chat()


def format_when(iso_timestamp):
    # Timestamps are UTC and the viewer's timezone is unknown, so show the day only
    when = datetime.fromisoformat(iso_timestamp)
    return "Today" if when.date() == datetime.utcnow().date() else when.strftime("%d %b")


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

/* Chat history: one row per saved chat, the open one highlighted */
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: 6px; margin-bottom: -8px; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button { justify-content: flex-start; }
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] .stButton button p {
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.88rem;
}
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"]:last-child .stButton button {
    justify-content: center; background: transparent; border-color: transparent; padding: 0 4px;
}
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: rgba(255, 255, 255, 0.3);
    border: 1px solid #ffffff;
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
[data-testid="stChatMessageContent"] h1, [data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3, [data-testid="stChatMessageContent"] h4 {
    color: #fff; font-size: 1.05rem; font-weight: 600; padding: 12px 0 4px;
}
[data-testid="stChatMessageContent"] strong { color: #fff; }
[data-testid="stChatMessageContent"] table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
[data-testid="stChatMessageContent"] th, [data-testid="stChatMessageContent"] td {
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 6px 10px;
    color: #fff;
    text-align: left;
}
[data-testid="stChatMessageContent"] th { background: rgba(255, 255, 255, 0.1); font-weight: 600; }

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

# There is no login, so each browser gets a random id kept in the URL (?uid=...).
# History is filtered by it, so visitors of the public app only ever see their own chats.
if "uid" not in st.query_params:
    st.query_params["uid"] = uuid.uuid4().hex
user_id = st.query_params["uid"]

if "session_id" not in st.session_state:
    # Fresh browser session (first visit or a page refresh): reopen the chat named in the URL
    st.session_state.messages = []
    st.session_state.session_id = None
    chat_id = st.query_params.get("chat")
    chat_tenant = get_session_tenant(chat_id, user_id) if chat_id else None
    if chat_tenant in tenants:
        open_session(chat_id, chat_tenant, user_id)
    else:
        st.query_params.pop("chat", None)

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
        key="tenant_picker",
    )
    tenant = tenants[tenant_id]
    st.write("")
    st.button("➕ New Chat", use_container_width=True, on_click=start_new_chat)

    st.markdown("### 📜 Chat History")
    sessions = list_sessions(user_id)
    if not sessions:
        st.caption("No saved chats yet.")
    for s in sessions:
        is_open = s["session_id"] == st.session_state.session_id
        open_col, delete_col = st.columns([5, 1], vertical_alignment="center")
        open_col.button(
            s["title"],
            key=f"open_{s['session_id']}",
            help=f"{tenants.get(s['tenant_id'], {}).get('tenant_name', s['tenant_id'])} · {format_when(s['updated_at'])}",
            type="primary" if is_open else "secondary",
            use_container_width=True,
            on_click=open_session,
            args=(s["session_id"], s["tenant_id"], user_id),
        )
        delete_col.button(
            "🗑️",
            key=f"delete_{s['session_id']}",
            help="Delete this chat",
            on_click=remove_session,
            args=(s["session_id"], user_id),
        )

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

# Switching organization starts a new chat (opening a saved chat sets tenant_id first, so it doesn't)
if st.session_state.get("tenant_id") != tenant_id:
    st.session_state.tenant_id = tenant_id
    start_new_chat()

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"].replace("$", "\\$"))

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt.replace("$", "\\$"))

    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(
                escape_dollars(
                    stream_reply(
                        api_key,
                        model,
                        st.session_state.messages,
                        build_system_prompt(tenant),
                        tenant.get("llm", {}).get("temperature", 0.7),
                    )
                )
            )
        except Exception as e:
            st.session_state.messages.pop()  # let the user retry the same message
            st.error(str(e))
            st.stop()

    # Store the raw text (it is sent back to the model as history); escaping happens at render time
    reply = reply.replace("\\$", "$")
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Save the turn only once it succeeded, so failed requests don't leave half-saved chats
    if not st.session_state.session_id:
        st.session_state.session_id = create_session(user_id, tenant_id, prompt)
        st.query_params["chat"] = st.session_state.session_id
    add_message(st.session_state.session_id, "user", prompt)
    add_message(st.session_state.session_id, "assistant", reply)
    st.rerun()  # refresh the sidebar so the new or updated chat shows up
