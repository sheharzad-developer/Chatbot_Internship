"""
Chat page: the multi-tenant chatbot (history, PDF uploads, web search, copy).

Calls a hosted API, so it also runs on Streamlit Community Cloud: Groq when GROQ_API_KEY is
set, otherwise Gemini. Tenant system prompts and temperatures are read from the same
backend/config/tenants/*.yaml files the FastAPI backend uses.

Part of the multipage app started with:  streamlit run streamlit_app/app.py
"""
import html
import io
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st
import streamlit.components.v1 as components
import yaml
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from llm import get_secret, stream_gemini, stream_groq
from theme import icon, page_header

ROOT = Path(__file__).resolve().parent.parent
TENANTS_DIR = ROOT / "backend" / "config" / "tenants"
HISTORY_LIMIT = 20



# Adapted from the guidance backend/agents/tool_aware_agent.py adds when a tenant has no tools
# connected. Without it the model role-plays the tools named in the tenant prompt and invents
# their results. (Web results and document excerpts, when present, arrive inside the message.)
NO_TOOLS_GUIDANCE = (
    "\n\nIMPORTANT: No internal databases or enterprise systems are connected. Apart from document "
    "excerpts and web search results included in a message, answer from your general knowledge. "
    "Never claim to use a tool, and never invent search results, internal company figures, "
    "policy or compliance checks. If a question needs "
    "internal company data, say you don't have access to it. When prices or costs would help "
    "(hotels, flights, trips, products), do include typical price ranges from general knowledge, "
    "label them as approximate, and suggest checking a live source for current prices."
)
FORMAT_GUIDANCE = (
    "\n\nFormat answers in clean Markdown: a one-line intro, short headings or numbered lists "
    "where they help, and tables only for real comparisons. Keep it concise."
)


RAG_GUIDANCE = (
    "\n\nThe user has uploaded documents. Some messages start with numbered excerpts from them: "
    "those excerpts are real and are your primary source. Answer from them and cite them inline "
    "like [1] or [2]. If the question is about the documents but they don't contain the answer, say "
    "so, then add general knowledge only if it helps, clearly labelled as such. If the question has "
    "nothing to do with the documents, just answer it normally without mentioning them or citing. "
    "When the answer needs a calculation from the documents (for example total years of experience "
    "from job dates), work it out from the dates, briefly show how you counted, and treat 'Present' "
    "as today's date: {today}."
)


DATE_GUIDANCE = (
    "\n\nToday's date is {today}. Your built-in knowledge ends around mid-2024. For anything more "
    "recent, rely on web search results when a message includes them; otherwise say your "
    "information may be out of date rather than guessing."
)
WEB_GUIDANCE = (
    "\n\nThis message includes web search results gathered just now: they are real and current. "
    "Use them for anything recent, cite them inline like [3], and prefer them over your built-in "
    "knowledge when they differ. If they don't answer the question, say so."
)


def today():
    return datetime.utcnow().strftime("%B %d, %Y")


def build_system_prompt(tenant, has_documents=False, has_web_results=False):
    prompt = tenant["system_prompt"] + NO_TOOLS_GUIDANCE + FORMAT_GUIDANCE + DATE_GUIDANCE.format(today=today())
    if has_documents:
        prompt += RAG_GUIDANCE.format(today=today())
    if has_web_results:
        prompt += WEB_GUIDANCE
    return prompt


def escape_dollars(chunks):
    """st.markdown treats text between $ signs as LaTeX, which garbles prices like "$200 – $400"."""
    for chunk in chunks:
        # Some models cite as 【1】; show the plain [1] the prompt asks for
        yield chunk.replace("$", "\\$").replace("【", "[").replace("】", "]")


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

-- Uploaded PDFs (RAG): split into page-tagged chunks, private to the uploading browser's user_id
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL,
    name         TEXT NOT NULL,
    pages        INTEGER NOT NULL,
    uploaded_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id  TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page    INTEGER NOT NULL,
    text    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id, uploaded_at);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id, seq);
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


def add_message(session_id, role, content, metadata=None):
    now = datetime.utcnow().isoformat()
    with history_db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (id, session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), session_id, role, content, now, json.dumps(metadata or {})),
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
            {"role": m["role"], "content": m["content"], **json.loads(m["metadata"])}
            for m in conn.execute(
                "SELECT m.role, m.content, m.metadata FROM chat_messages m "
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


# --- RAG over uploaded PDFs -------------------------------------------------------------------
# Retrieval uses TF-IDF (scikit-learn) rather than the backend's sentence-transformers: it needs
# no PyTorch, so it fits in Streamlit Community Cloud's memory limits.
MAX_PDF_MB = 10
MAX_PDF_PAGES = 300
MAX_DOCS_PER_USER = 10
CHUNK_CHARS = 900
CHUNK_OVERLAP = 150
TOP_K = 4
MIN_SCORE = 0.05
# Documents up to this size (a resume, a short report) are sent whole once a question matches them:
# answers like "total years of experience" need every part of the document, not the top few chunks
FULL_TEXT_CHARS = 15000
SUMMARY_REQUEST = re.compile(
    r"\b(summar\w*|overview|key points|main points|tl;?dr|outline)\b|what('s| is) (this|the|my) (pdf|document|doc|file|paper|report)",
    re.I,
)
DOC_QUESTION = re.compile(r"\b(pdf|document|doc|file|upload\w*|paper|report|attached|resume|résumé|cv)\b", re.I)


def extract_pdf_pages(data):
    """Return [(page_number, text)] for pages that contain text."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("it is password-protected")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"it has more than {MAX_PDF_PAGES} pages")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            pages.append((number, text))
    return pages, len(reader.pages)


def chunk_pages(pages):
    """Split each page into overlapping chunks that end on word boundaries."""
    chunks = []
    for number, text in pages:
        start = 0
        while start < len(text):
            end = min(start + CHUNK_CHARS, len(text))
            space = text.rfind(" ", start, end)
            if end < len(text) and space > start:
                end = space
            chunks.append((number, text[start:end].strip()))
            if end >= len(text):
                break
            start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def add_document(user_id, name, data):
    """Store a PDF's chunks. Returns an error message, or None on success."""
    if len(data) > MAX_PDF_MB * 1024 * 1024:
        return f"{name}: skipped, larger than {MAX_PDF_MB} MB"
    if len(list_documents(user_id)) >= MAX_DOCS_PER_USER:
        return f"{name}: skipped, you can keep up to {MAX_DOCS_PER_USER} documents (delete one first)"
    try:
        pages, page_count = extract_pdf_pages(data)
    except Exception as e:
        return f"{name}: couldn't read it ({e})"
    if not pages:
        return f"{name}: no text found. Scanned PDFs (images of text) aren't supported"
    doc_id = str(uuid.uuid4())
    with history_db() as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, user_id, name, pages, uploaded_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, user_id, name, page_count, datetime.utcnow().isoformat()),
        )
        conn.executemany(
            "INSERT INTO document_chunks (doc_id, page, text) VALUES (?, ?, ?)",
            [(doc_id, page, text) for page, text in chunk_pages(pages)],
        )
    return None


def list_documents(user_id):
    with history_db() as conn:
        return conn.execute(
            "SELECT doc_id, name, pages FROM documents WHERE user_id = ? ORDER BY uploaded_at DESC", (user_id,)
        ).fetchall()


def delete_document(doc_id, user_id):
    with history_db() as conn:
        conn.execute("DELETE FROM documents WHERE doc_id = ? AND user_id = ?", (doc_id, user_id))


@st.cache_resource(max_entries=100, show_spinner=False)
def build_index(user_id, doc_ids):
    """TF-IDF index over one user's chunks; cached until their set of documents changes."""
    with history_db() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                "SELECT d.doc_id, d.name, c.page, c.text FROM document_chunks c "
                "JOIN documents d ON d.doc_id = c.doc_id "
                "WHERE d.user_id = ? ORDER BY d.uploaded_at DESC, c.seq",
                (user_id,),
            )
        ]
    if not rows:
        return None
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(r["text"] for r in rows)
    except ValueError:  # every chunk was stop words / empty vocabulary
        return None
    return vectorizer, matrix, rows


def retrieve(user_id, docs, prompt, previous_prompt=None):
    """Chunks from the user's documents relevant to the prompt, as [{name, page, text}]."""
    index = build_index(user_id, tuple(d["doc_id"] for d in docs))
    if not index:
        return []
    vectorizer, matrix, rows = index

    if sum(len(r["text"]) for r in rows) <= FULL_TEXT_CHARS:
        # Small library (a resume, a few short PDFs): send everything, so no question can miss.
        # Only the passages the answer actually cites are shown as sources (see cited_sources).
        hits = rows
    elif SUMMARY_REQUEST.search(prompt):
        # A summary needs the whole document, not keyword matches: take chunks spread evenly
        # across the newest document's pages
        newest = [r for r in rows if r["doc_id"] == rows[0]["doc_id"]]
        step = max(1, len(newest) // (TOP_K + 2))
        hits = newest[::step][: TOP_K + 2]
    else:
        def search(query):
            scores = linear_kernel(vectorizer.transform([query]), matrix).ravel()
            return [rows[i] for i in scores.argsort()[::-1][:TOP_K] if scores[i] >= MIN_SCORE]

        hits = search(prompt)
        if not hits and previous_prompt:
            # Short follow-ups ("and how many can I carry over?") match better with the last question
            hits = search(f"{previous_prompt} {prompt}")
        if not hits and DOC_QUESTION.search(prompt):
            # "What's in the file?"-style questions have no keywords to match: use the newest doc's start
            hits = [r for r in rows if r["doc_id"] == rows[0]["doc_id"]][:TOP_K]

    # If the matched documents are short, send them whole instead of just the matching chunks
    matched = {h["doc_id"] for h in hits}
    whole = [r for r in rows if r["doc_id"] in matched]
    if hits and sum(len(r["text"]) for r in whole) <= FULL_TEXT_CHARS:
        hits = whole
    return [{"name": h["name"], "page": h["page"], "text": h["text"]} for h in hits]


def with_context(prompt, sources):
    """The user message actually sent to the model: document excerpts and web results, then the question."""
    if not sources:
        return prompt
    docs, web = [], []
    for i, s in enumerate(sources, start=1):
        if s.get("type") == "web":
            web.append(f"[{i}] {s['title']} ({s['url']})\n{s['text']}")
        else:
            docs.append(f"[{i}] ({s['name']}, page {s['page']})\n{s['text']}")
    parts = []
    if docs:
        parts.append("Excerpts from my uploaded documents:\n\n" + "\n\n".join(docs))
    if web:
        parts.append(f"Web search results (searched {today()}):\n\n" + "\n\n".join(web))
    return "\n\n".join(parts) + f"\n\n---\nMy question: {prompt}"


# --- Web search (Tavily) -------------------------------------------------------------------------
TAVILY_URL = "https://api.tavily.com/search"
WEB_RESULTS = 5

ROUTER_PROMPT = """You decide whether a chatbot must search the web to answer the user's latest message.
Today's date is {today}. The chatbot's built-in knowledge ends around mid-2024.

SEARCH when the answer depends on recent information (after mid-2024) or facts that change over time:
news, sports results, elections, who currently holds a role, live prices or exchange rates, weather,
new product releases, or anything asking for the "latest", "current", "today" or "this year".
Any question about the outcome of an event in 2024 or later (a tournament, election, award, launch)
needs SEARCH, even if you think you know about the event: your knowledge of it may be outdated or
only a forecast.

NO_SEARCH for: timeless general knowledge, explanations, coding, maths, writing, advice, greetings,
chit-chat, or questions about the user's own uploaded documents{docs_note}.

Examples:
"Who won the 2026 Champions League final?" -> SEARCH: 2026 Champions League final winner
"What's the weather in Karachi?" -> SEARCH: Karachi weather today
"Who won the 2018 FIFA World Cup?" -> NO_SEARCH
"How does photosynthesis work?" -> NO_SEARCH

Reply with exactly one line, nothing else:
SEARCH: <short standalone search query; resolve follow-ups like "who won it?" using the conversation>
or
NO_SEARCH"""


def plan_web_search(messages, docs):
    """Ask a small, fast model whether the latest message needs a web search. Returns a query or None."""
    docs_note = f" (they uploaded: {', '.join(d['name'] for d in docs)})" if docs else ""
    transcript = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in messages[-5:])
    reply = "".join(
        stream_reply(
            api_key,
            router_model,
            [{"role": "user", "content": f"Conversation:\n{transcript}\n\nDecide for the last user message."}],
            ROUTER_PROMPT.format(today=today(), docs_note=docs_note),
            0,
        )
    )
    if "NO_SEARCH" in reply.upper():
        return None
    # Take the first query only; small models sometimes repeat the line without a line break
    match = re.search(r"SEARCH:\s*(.+?)(?=\s*SEARCH:|\n|$)", reply)
    return match.group(1).strip().strip('"') if match else None


def web_search(query):
    """Top web results from Tavily, as [{type, title, url, text}]."""
    response = httpx.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {tavily_key}"},
        json={"query": query, "max_results": WEB_RESULTS, "search_depth": "basic"},
        timeout=30,
    )
    response.raise_for_status()
    return [
        {"type": "web", "title": r.get("title") or r["url"], "url": r["url"], "text": (r.get("content") or "")[:800]}
        for r in response.json().get("results", [])
    ]


COPY_BUTTON = """
<style>
  body { margin: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
  button {
    display: inline-flex; align-items: center; gap: 6px;
    background: transparent; color: #6B7280; cursor: pointer;
    border: 1px solid #E3E6EB; border-radius: 8px;
    padding: 4px 10px; font-size: 12.5px; transition: background 0.15s, color 0.15s;
  }
  button:hover { background: #F5F6F8; color: #1F2430; }
  button.done { color: #2F8F63; border-color: #BFE3D0; }
</style>
<button id="copy" title="Copy this answer"></button>
<script>
  const text = __TEXT__;
  const button = document.getElementById("copy");
  const svg = (paths) => '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + paths + '</svg>';
  const COPY = svg('<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>') + "Copy";
  const COPIED = svg('<path d="M20 6 9 17l-5-5"/>') + "Copied";
  button.innerHTML = COPY;
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (e) {
      // Older browsers / blocked clipboard API: copy through a hidden textarea instead
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    button.innerHTML = COPIED;
    button.classList.add("done");
    setTimeout(() => { button.innerHTML = COPY; button.classList.remove("done"); }, 1500);
  });
</script>
"""


def copy_button(text):
    """A small button that copies the answer's Markdown to the clipboard."""
    # json.dumps makes a safe JS string; "</" is escaped so the text can't close the <script> tag
    components.html(COPY_BUTTON.replace("__TEXT__", json.dumps(text).replace("</", "<\\/")), height=34)


def cited_sources(reply, sources):
    """
    Number every source as it was given to the model. Document excerpts are kept only when the reply
    cites them as [n] (also [1, 2]); web results are always kept, since a search ran for this answer.
    """
    cited = {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", reply) for n in re.findall(r"\d+", group)}
    return [dict(s, n=i) for i, s in enumerate(sources, start=1) if i in cited or s.get("type") == "web"]


def clean_snippet(text, limit=220):
    """Web page text arrives with table pipes, headings and ad labels; make it a readable one-liner."""
    text = re.sub(r"\b(Advertisement|Image \d+:?)\b", " ", text)
    text = re.sub(r"[|#*_`>\[\]]+", " ", text)
    text = re.sub(r"-{3,}", " ", text)
    text = " ".join(text.split())
    text = text[:limit] + ("…" if len(text) > limit else "")
    return text.replace("$", "\\$")


def show_sources(sources, search_query=None):
    if not sources:
        return
    docs = [s for s in sources if s.get("type") != "web"]
    web = [s for s in sources if s.get("type") == "web"]
    if docs:
        with st.expander(f"Sources ({len(docs)})", icon=":material/description:"):
            for i, s in enumerate(docs, start=1):
                snippet = s["text"][:300] + ("…" if len(s["text"]) > 300 else "")
                st.markdown(f"**[{s.get('n', i)}] {s['name']} · page {s['page']}**  \n{snippet}".replace("$", "\\$"))
    if web:
        with st.expander(f"Web sources ({len(web)})", icon=":material/public:"):
            if search_query:
                st.caption(f"Searched: {search_query}")
            for i, s in enumerate(web, start=1):
                title = s["title"].replace("[", "(").replace("]", ")")
                # Link only plain http(s) URLs, so a result can't inject a javascript: link
                url = s["url"].replace(")", "%29").replace(" ", "%20")
                heading = f"[{title}]({url})" if re.match(r"https?://", url) else f"{title} ({s['url']})"
                site = re.sub(r"^https?://(www\.)?", "", s["url"]).split("/")[0]
                st.markdown(f"**[{s.get('n', i)}] {heading}**  \n<small>{html.escape(site)}</small>".replace("$", "\\$"),
                            unsafe_allow_html=True)
                st.caption(clean_snippet(s["text"]))


@st.cache_data
def load_tenants():
    tenants = {}
    for path in sorted(TENANTS_DIR.glob("*.yaml")):
        with open(path) as f:
            config = yaml.safe_load(f)
        tenants[config["tenant_id"]] = config
    return tenants


WELCOME = (
    "**Welcome!** Pick an organization in the sidebar and ask me anything.\n\n"
    "- Upload PDFs and ask questions about them, with cited pages\n"
    "- Questions about recent events are answered with a web search\n"
    "- Your chats are saved in the sidebar"
)


def ingest_uploads(user_id):
    """file_uploader callback: store the new PDFs, then swap in a fresh, empty uploader."""
    files = st.session_state.get(f"uploader_{st.session_state.uploader_n}") or []
    added, errors = [], []
    for f in files:
        error = add_document(user_id, f.name, f.getvalue())
        (errors.append(error) if error else added.append(f.name))
    st.session_state.upload_report = (added, errors)
    st.session_state.uploader_n += 1

st.markdown(
    page_header("AI Chatbot", "Multi-tenant assistant with document Q&amp;A and web search", "chat"),
    unsafe_allow_html=True,
)

if get_secret("GROQ_API_KEY"):
    api_key = get_secret("GROQ_API_KEY")
    model = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
    # The "does this need a web search?" check runs on every message, so use a smaller, faster model
    router_model = get_secret("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b")
    provider = "Groq"
    stream_reply = stream_groq
else:
    api_key = get_secret("GEMINI_API_KEY")
    model = router_model = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
    provider = "Gemini"
    stream_reply = stream_gemini
tavily_key = get_secret("TAVILY_API_KEY")
tenants = load_tenants()

# There is no login, so each browser gets a random id kept in the URL (?uid=...).
# History is filtered by it, so visitors of the public app only ever see their own chats.
# Switching pages can drop the query string, so the id is also kept in the session and put back.
if "uid" not in st.query_params:
    st.query_params["uid"] = st.session_state.get("uid") or uuid.uuid4().hex
user_id = st.session_state.uid = st.query_params["uid"]
if st.session_state.get("session_id") and "chat" not in st.query_params:
    st.query_params["chat"] = st.session_state.session_id

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
    st.markdown("### Organizations")
    tenant_id = st.radio(
        "Organization",
        options=list(tenants),
        format_func=lambda t: tenants[t]["tenant_name"],
        captions=[
            f"{tenants[t].get('description', '')}  \n"
            f"Temperature {tenants[t].get('llm', {}).get('temperature', 0.7)} · {provider}"
            for t in tenants
        ],
        label_visibility="collapsed",
        key="tenant_picker",
    )
    tenant = tenants[tenant_id]
    st.write("")
    st.button("New chat", icon=":material/add:", use_container_width=True, on_click=start_new_chat)
    web_enabled = st.toggle(
        ":material/language: Web search",
        value=bool(tavily_key),
        disabled=not tavily_key,
        help="Searches the web automatically when a question needs current information (news, scores, prices...).",
    )
    if not tavily_key:
        st.caption("Add `TAVILY_API_KEY` to the app's secrets to enable web search (free at tavily.com).")

    st.markdown("### Documents")
    st.session_state.setdefault("uploader_n", 0)
    st.file_uploader(
        f"Upload PDFs (up to {MAX_PDF_MB} MB each) and ask questions about them",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_n}",
        on_change=ingest_uploads,
        args=(user_id,),
    )
    added, errors = st.session_state.pop("upload_report", ([], []))
    if added:
        st.success("Added: " + ", ".join(added))
    for error in errors:
        st.warning(error)
    docs = list_documents(user_id)
    for d in docs:
        name_col, delete_col = st.columns([5, 1], vertical_alignment="center")
        name_col.markdown(
            f"<div class='doc-name'>{icon('file', 16)}<span>{html.escape(d['name'])}<br>"
            f"<small>{d['pages']} pages</small></span></div>",
            unsafe_allow_html=True,
        )
        delete_col.button(
            ":material/delete:",
            key=f"delete_doc_{d['doc_id']}",
            help="Delete this document",
            on_click=delete_document,
            args=(d["doc_id"], user_id),
        )

    st.markdown("### Chat history")
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
            ":material/delete:",
            key=f"delete_{s['session_id']}",
            help="Delete this chat",
            on_click=remove_session,
            args=(s["session_id"], user_id),
        )

st.markdown(
    f'<div class="chat-header"><div class="chat-title">{html.escape(tenant["tenant_name"])}</div>'
    f'<div class="model-pill">{provider} · {html.escape(model)}</div></div>',
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
        if message.get("search_note"):
            st.caption(message["search_note"])
        st.markdown(message["content"].replace("$", "\\$"))
        show_sources(message.get("sources"), message.get("search_query"))
        if message["role"] == "assistant":
            copy_button(message["content"])

placeholder = "Ask about your PDFs, or anything else..." if docs else "Ask me anything..."
if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt.replace("$", "\\$"))

    previous = [m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"]
    sources = retrieve(user_id, docs, prompt, previous[-1] if previous else None) if docs else []

    with st.chat_message("assistant"):
        search_query, web_results, search_note = None, [], None
        if web_enabled:
            try:
                with st.spinner("Thinking..."):
                    search_query = plan_web_search(st.session_state.messages, docs)
                if search_query:
                    with st.spinner(f"Searching the web: {search_query}"):
                        web_results = web_search(search_query)
            except Exception as e:
                # Search is a bonus: if it fails, answer without it rather than failing the message
                status = getattr(getattr(e, "response", None), "status_code", None)
                reason = {401: "invalid Tavily API key", 429: "rate limit reached"}.get(status, e.__class__.__name__)
                search_note = f":material/warning: Web search unavailable ({reason}), so this answer doesn't include current information."
                st.caption(search_note)
                search_query, web_results = None, []
        sources = sources + web_results

        # Excerpts and web results go only into this request; history keeps the plain question
        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        api_messages.append({"role": "user", "content": with_context(prompt, sources)})
        try:
            reply = st.write_stream(
                escape_dollars(
                    stream_reply(
                        api_key,
                        model,
                        api_messages,
                        build_system_prompt(tenant, has_documents=bool(docs), has_web_results=bool(web_results)),
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
    sources = cited_sources(reply, sources)
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources, "search_query": search_query, "search_note": search_note}
    )

    # Save the turn only once it succeeded, so failed requests don't leave half-saved chats
    if not st.session_state.session_id:
        st.session_state.session_id = create_session(user_id, tenant_id, prompt)
        st.query_params["chat"] = st.session_state.session_id
    add_message(st.session_state.session_id, "user", prompt)
    add_message(
        st.session_state.session_id, "assistant", reply,
        {"sources": sources, "search_query": search_query, "search_note": search_note},
    )
    st.rerun()  # refresh the sidebar so the new or updated chat shows up
