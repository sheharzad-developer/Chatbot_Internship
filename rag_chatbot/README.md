# 📚 Domain-Specific RAG Chatbot

Ask questions about your own PDFs. The chatbot answers **only** from them; questions outside
the documents get: *"The resource has no information about this."*

Fully local: embeddings with `all-MiniLM-L6-v2`, vectors in **ChromaDB**, answers from
**Llama 3.2 via Ollama**.

## Files

| File | What it does |
|---|---|
| `documents/` | Put your domain PDF(s) here |
| `ingest.py` | Reads the PDFs, splits them into chunks, embeds them, saves the model to `models/` and the vectors to `chroma_db/` |
| `rag.py` | Shared logic: settings, chunking, retrieval, the prompt, and calling Ollama |
| `app.py` | The Streamlit chat interface |

## Run it

```bash
# 1. Install (from the repo root, once)
python3 -m venv .venv
.venv/bin/pip install -r rag_chatbot/requirements.txt
ollama pull llama3.2          # and keep Ollama running

# 2. Add your PDF(s) to rag_chatbot/documents/, then build the vector store
cd rag_chatbot
../.venv/bin/python ingest.py # creates models/ and chroma_db/

# 3. Start the chatbot
../.venv/bin/streamlit run app.py
```

Optionally name your domain so the app and prompt use it:

```bash
RAG_DOMAIN="Pakistan Labour Law" ../.venv/bin/streamlit run app.py
```

Re-run `ingest.py` whenever you add, change or remove PDFs.

## How it stays on-topic

1. **Similarity filter.** Each question is embedded and compared with the chunks. If no chunk reaches
   `MIN_SIMILARITY` (0.25), the app replies *"The resource has no information about this."* without
   calling the LLM. In testing, on-topic questions scored 0.31–0.61 and off-topic ones −0.05–0.10.
2. **Strict prompt.** The LLM is told to answer only from the retrieved passages and to reply with
   exactly that sentence if they don't contain the answer (e.g. a related but uncovered topic).

Each answer shows its sources (file, page and similarity score).
