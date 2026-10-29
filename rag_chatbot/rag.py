"""
Shared RAG logic: settings, PDF chunking, retrieval from ChromaDB, and answering with Ollama.

ingest.py uses it to build the vector store; app.py uses it to answer questions.
"""
import json
import logging
import os
from pathlib import Path

import chromadb
import httpx
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Telemetry is disabled below, but chromadb 0.6.3 still logs a harmless "Failed to send telemetry
# event" with newer posthog versions; hide it so it doesn't look like an error
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# --- Settings ------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"   # put your domain PDFs here
CHROMA_DIR = BASE_DIR / "chroma_db"      # created by ingest.py
MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"  # embedding model, saved by ingest.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"

# Name of your PDF's domain, shown in the app and used in the prompt, e.g. "Pakistan Labour Law"
DOMAIN = os.getenv("RAG_DOMAIN", "the uploaded documents")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks
TOP_K = 4              # chunks retrieved per question
# Cosine similarity a chunk needs to count as relevant. Below it, the question is treated as
# outside the domain and answered with NO_INFO without calling the LLM.
MIN_SIMILARITY = 0.25

NO_INFO = "The resource has no information about this."

SYSTEM_PROMPT = f"""You are a question-answering assistant for {DOMAIN}.
Answer ONLY using the context passages provided with each question.
- If the context does not contain the answer, reply with exactly this sentence and nothing else:
  {NO_INFO}
- Never use outside knowledge, and never guess.
- Keep answers clear and concise. Use bullet points for lists.
- Mention the page numbers you used, like (page 3)."""


# --- Ingestion helpers ---------------------------------------------------------------------------
def read_pdf(path):
    """Return [(page_number, text)] for the PDF's pages that contain text."""
    pages = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        if text:
            pages.append((number, text))
    return pages


def chunk_text(text):
    """Split text into overlapping chunks that end on word boundaries."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        space = text.rfind(" ", start, end)
        if end < len(text) and space > start:
            end = space
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def load_embedder():
    """Load the embedding model saved by ingest.py (no download needed at question time)."""
    if not MODEL_DIR.exists():
        raise FileNotFoundError("Embedding model not found. Run `python ingest.py` first.")
    return SentenceTransformer(str(MODEL_DIR))


def chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def load_collection():
    """Open the vector store built by ingest.py."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("Vector store not found. Run `python ingest.py` first.")
    return chroma_client().get_collection(COLLECTION_NAME)


# --- Question answering --------------------------------------------------------------------------
def retrieve(question, embedder, collection, k=TOP_K):
    """The k most similar chunks, as [{text, source, page, similarity}], most similar first."""
    embedding = embedder.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=embedding, n_results=k, include=["documents", "metadatas", "distances"])
    return [
        {"text": text, "source": meta["source"], "page": meta["page"], "similarity": 1 - distance}
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


def build_prompt(question, chunks):
    context = "\n\n".join(f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in chunks)
    return f"Context:\n{context}\n\nQuestion: {question}"


def stream_ollama(prompt):
    """Yield the answer text chunk by chunk from the local Ollama server."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "options": {"temperature": 0},  # stick to the context, no creativity
        "stream": True,
    }
    with httpx.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload, timeout=300) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text[:300]}")
        for line in response.iter_lines():
            if line:
                text = json.loads(line).get("message", {}).get("content")
                if text:
                    yield text


def answer(question, embedder, collection):
    """
    Returns (answer, sources). `answer` is NO_INFO when nothing relevant was found,
    otherwise a generator streaming the LLM's answer.
    """
    sources = [c for c in retrieve(question, embedder, collection) if c["similarity"] >= MIN_SIMILARITY]
    if not sources:
        return NO_INFO, []
    return stream_ollama(build_prompt(question, sources)), sources


def is_no_info(reply):
    """The model was told to reply with NO_INFO when the context lacks the answer."""
    return NO_INFO.lower().rstrip(".") in reply.lower()


def ollama_ready():
    """True if Ollama is running and the model is pulled."""
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
    except httpx.HTTPError:
        return False
    return any(m["name"].split(":")[0] == OLLAMA_MODEL.split(":")[0] for m in tags.get("models", []))
