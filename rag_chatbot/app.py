"""
Domain-specific RAG chatbot.

    streamlit run app.py

Answers only from the PDFs indexed by ingest.py. Questions outside them get:
"The resource has no information about this."
"""
import streamlit as st

from rag import (
    DOMAIN, NO_INFO, OLLAMA_MODEL, answer, is_no_info, load_collection, load_embedder, ollama_ready,
)

st.set_page_config(page_title="Domain RAG Chatbot", page_icon="📚")


@st.cache_resource(show_spinner="Loading the embedding model and vector store...")
def load_resources():
    collection = load_collection()
    sources = sorted({m["source"] for m in collection.get(include=["metadatas"])["metadatas"]})
    return load_embedder(), collection, sources


def md(text):
    """Streamlit reads text between $ signs as LaTeX math, which garbles amounts like "$45 ... $120"."""
    return text.replace("$", "\\$")


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{md(s['source'])} · page {s['page']}** (similarity {s['similarity']:.2f})")
            st.caption(md(s["text"][:400] + ("…" if len(s["text"]) > 400 else "")))


st.title("📚 Domain RAG Chatbot")
st.caption(f"Ask questions about **{DOMAIN}**. Answers come only from the indexed PDFs.")

try:
    embedder, collection, pdf_names = load_resources()
except Exception as e:
    st.error(f"{e}\n\nPut your PDF(s) in `rag_chatbot/documents/`, then run `python ingest.py`.")
    st.stop()

with st.sidebar:
    st.header("Knowledge base")
    st.markdown(f"**Domain:** {DOMAIN}")
    st.markdown(f"**Chunks indexed:** {collection.count()}")
    for name in pdf_names:
        st.markdown(f"📄 {name}")
    st.markdown(f"**LLM:** {OLLAMA_MODEL} (Ollama, local)")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not ollama_ready():
    st.error(f"Ollama isn't running or `{OLLAMA_MODEL}` isn't pulled. Run `ollama serve` and `ollama pull {OLLAMA_MODEL}`.")
    st.stop()

st.session_state.setdefault("messages", [])
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(md(message["content"]))
        show_sources(message.get("sources"))

if question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(md(question))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                reply, sources = answer(question, embedder, collection)
            if reply != NO_INFO:
                # Show escaped text while streaming, keep the raw text for the chat history
                reply = st.write_stream(md(chunk) for chunk in reply).replace("\\$", "$")
            if reply == NO_INFO or is_no_info(reply):
                # Retrieval found nothing, or the model said the context lacks the answer
                reply, sources = NO_INFO, []
        except Exception as e:
            st.session_state.messages.pop()
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
    st.rerun()
