"""
Domain RAG page: answers only from the PDFs in documents/ (indexed by ingest.py).
Questions outside them get: "The resource has no information about this."

LLM: Ollama when it's running locally, otherwise Groq (or Gemini) with the chat page's key.
"""
import html

import streamlit as st

from llm import get_secret, stream_gemini, stream_groq
from theme import icon, page_header
from rag import (
    DOMAIN, NO_INFO, OLLAMA_MODEL, answer, build_knowledge_base, indexed_pdfs, is_no_info,
    load_collection, ollama_ready, pdf_files, stream_ollama,
)


def md(text):
    """Streamlit reads text between $ signs as LaTeX math, which garbles amounts like "$45 ... $120"."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_knowledge_base():
    collection = load_collection()
    return collection, indexed_pdfs(collection)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_available():
    return ollama_ready()


def pick_llm():
    """(label, stream_llm) for the first available LLM: local Ollama, then Groq, then Gemini."""
    if ollama_available():
        return f"{OLLAMA_MODEL} (Ollama, local)", lambda system, prompt: stream_ollama(prompt)
    for name, stream, key_name, model_name, default_model in [
        ("Groq", stream_groq, "GROQ_API_KEY", "GROQ_MODEL", "openai/gpt-oss-120b"),
        ("Gemini", stream_gemini, "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash"),
    ]:
        key = get_secret(key_name)
        if key:
            model = get_secret(model_name, default_model)
            return f"{model} ({name})", lambda system, prompt: stream(
                key, model, [{"role": "user", "content": prompt}], system, 0
            )
    return None, None


def rebuild():
    with st.status("Building the knowledge base...", expanded=True) as status:
        try:
            stats = build_knowledge_base(log=st.write)
        except ValueError as e:
            status.update(label=str(e), state="error")
            return False
        status.update(label=f"Indexed {stats['chunks']} chunks from {stats['pdfs']} PDF(s)", state="complete")
    load_knowledge_base.clear()
    return True


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", icon=":material/description:"):
        for s in sources:
            st.markdown(f"**{md(s['source'])} · page {s['page']}** (similarity {s['similarity']:.2f})")
            st.caption(md(s["text"][:400] + ("…" if len(s["text"]) > 400 else "")))


st.markdown(
    page_header("Domain RAG", f"Answers about {html.escape(DOMAIN)}, only from the indexed PDFs", "book"),
    unsafe_allow_html=True,
)

pdfs_on_disk = [p.name for p in pdf_files()]
try:
    collection, pdf_names = load_knowledge_base()
except Exception:
    if not pdfs_on_disk:
        st.info("No documents yet. Put your domain PDF(s) in `streamlit_app/documents/`, then run "
                "`python ingest.py` (or reload this page).")
        st.stop()
    # PDFs are there but no index yet (e.g. a fresh Streamlit Cloud deploy): build it now
    if not rebuild():
        st.stop()
    collection, pdf_names = load_knowledge_base()

llm_label, stream_llm = pick_llm()

with st.sidebar:
    st.markdown("### Knowledge base")
    st.markdown(f"**Domain:** {DOMAIN}  \n**Chunks indexed:** {collection.count()}")
    for name in pdf_names:
        st.markdown(f"<div class='doc-name'>{icon('file', 16)}<span>{html.escape(name)}</span></div>",
                    unsafe_allow_html=True)
    if sorted(pdfs_on_disk) != pdf_names:
        st.warning("`documents/` has changed since the last build.", icon=":material/sync_problem:")
    if st.button("Rebuild index", icon=":material/refresh:", use_container_width=True,
                 help="Re-read every PDF in documents/ and rebuild the knowledge base"):
        if rebuild():
            st.rerun()
    st.markdown(f"**LLM:** {llm_label or 'none available'}")
    if st.button("Clear chat", icon=":material/delete_sweep:", use_container_width=True, key="clear_domain_chat"):
        st.session_state.domain_messages = []
        st.rerun()

if not stream_llm:
    st.error(f"No LLM available. Start Ollama (`ollama serve`, `ollama pull {OLLAMA_MODEL}`) "
             "or set `GROQ_API_KEY` in `.streamlit/secrets.toml`.")
    st.stop()

# Kept separate from the Chat page's messages
st.session_state.setdefault("domain_messages", [])
for message in st.session_state.domain_messages:
    with st.chat_message(message["role"]):
        st.markdown(md(message["content"]))
        show_sources(message.get("sources"))

if question := st.chat_input("Ask a question about the documents...", key="domain_input"):
    st.session_state.domain_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                reply, sources = answer(question, collection, stream_llm)
            if reply != NO_INFO:
                # Show escaped text while streaming, keep the raw text for the chat history
                reply = st.write_stream(md(chunk) for chunk in reply).replace("\\$", "$")
            if reply == NO_INFO or is_no_info(reply):
                # Retrieval found nothing, or the model said the context lacks the answer
                reply, sources = NO_INFO, []
        except Exception as e:
            st.session_state.domain_messages.pop()
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.session_state.domain_messages.append({"role": "assistant", "content": reply, "sources": sources})
    st.rerun()
