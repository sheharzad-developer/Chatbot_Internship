# 🤖 AI Chatbot (Streamlit)

One Streamlit app with two pages, picked from the sidebar:

| Page | What it does |
|---|---|
| **💬 Chat** | Multi-tenant chatbot on Groq (or Gemini): chat history, PDF uploads with cited answers, automatic web search, copy button |
| **📚 Domain RAG** | Answers **only** from the PDFs in `documents/`; anything else gets *"The resource has no information about this."* |

## Files

| File | Purpose |
|---|---|
| `app.py` | Entry point: page navigation and the shared theme |
| `chat_page.py` | 💬 Chat page |
| `domain_page.py` | 📚 Domain RAG page |
| `rag.py` | Domain RAG logic: chunking, ChromaDB, retrieval, prompt, Ollama |
| `ingest.py` | Builds the Domain RAG knowledge base from `documents/` |
| `llm.py` / `theme.py` | Groq/Gemini streaming and secrets / purple theme, shared by both pages |
| `documents/` | Your domain PDFs |
| `chroma_db/`, `models/` | Created by `ingest.py` (vector store and the saved embedding model) |
| `data/` | Chat history and uploaded PDFs for the Chat page (SQLite) |

## Run it

```bash
# From the repo root, once
python3 -m venv .venv
.venv/bin/pip install -r streamlit_app/requirements.txt

# Secrets: .streamlit/secrets.toml
#   GROQ_API_KEY   = "gsk_..."    # Chat page, and Domain RAG when Ollama isn't running
#   TAVILY_API_KEY = "tvly-..."   # optional: web search on the Chat page

# Domain RAG: add your PDF(s), then build the knowledge base
cp ~/my-domain.pdf streamlit_app/documents/
cd streamlit_app
../.venv/bin/python ingest.py

# Start the app
../.venv/bin/streamlit run app.py
```

Name your domain so the Domain RAG page and its prompt use it:

```bash
RAG_DOMAIN="Pakistan Labour Law" ../.venv/bin/streamlit run app.py
```

Re-run `ingest.py` (or press **🔄 Rebuild knowledge base** on the page) whenever you add,
change or remove PDFs.

## Domain RAG details

- **Embeddings:** `all-MiniLM-L6-v2` as ONNX (ChromaDB's built-in embedder, no PyTorch), downloaded
  from Hugging Face on the first ingest and saved to `models/`.
- **LLM:** Llama 3.2 via **Ollama** when it's running locally; otherwise **Groq** (then Gemini) with
  the same key as the Chat page, so the page also works on Streamlit Community Cloud.
- **Staying on-topic:**
  1. If no chunk reaches a cosine similarity of `0.25`, the reply is *"The resource has no
     information about this."* without calling the LLM.
  2. The LLM is told to answer only from the retrieved passages, and to give that exact sentence
     when they don't contain the answer.
- **Streamlit Community Cloud:** commit your PDFs in `documents/`; the page builds the knowledge base
  automatically on first load (`chroma_db/` and `models/` are gitignored).
