"""
Domain RAG logic: settings, PDF chunking, building the ChromaDB knowledge base, retrieval,
and answering with Ollama (local) or Groq (hosted).

ingest.py calls build_knowledge_base(); domain_page.py uses the rest.
Nothing here imports Streamlit, so `python ingest.py` runs on its own.
"""
import json
import logging
import os
from pathlib import Path

import chromadb
import httpx
from chromadb.config import Settings
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from pypdf import PdfReader

# Telemetry is disabled below, but chromadb 0.6.3 still logs a harmless "Failed to send telemetry
# event" with newer posthog versions; hide it so it doesn't look like an error
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

# --- Settings ------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"   # put your domain PDFs here
CHROMA_DIR = BASE_DIR / "chroma_db"      # created by ingest.py
MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"  # embedding model, saved on first ingest
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


HF_MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/"
# File chromadb expects in models/.../onnx/  ->  its path in the Hugging Face repo
MODEL_FILES = {
    "model.onnx": "onnx/model.onnx",
    "config.json": "config.json",
    "special_tokens_map.json": "special_tokens_map.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "tokenizer.json": "tokenizer.json",
    "vocab.txt": "vocab.txt",
}


class ProjectMiniLM(ONNXMiniLM_L6_V2):
    """
    all-MiniLM-L6-v2 as the ONNX model built into ChromaDB: same embeddings as sentence-transformers,
    without PyTorch (so it fits on Streamlit Community Cloud). Saved inside the project, not ~/.cache.
    """
    DOWNLOAD_PATH = MODEL_DIR

    def __init__(self):
        # By default onnxruntime picks every accelerator it finds; on Macs that includes CoreML,
        # which fails on this model. The CPU provider works everywhere and is fast enough here.
        super().__init__(preferred_providers=["CPUExecutionProvider"])

    def _download_model_if_not_exists(self):
        # chromadb's own download server can be very slow (~10 KB/s on some networks), so fetch the
        # same ONNX export from Hugging Face's CDN instead, file by file
        folder = Path(self.DOWNLOAD_PATH) / self.EXTRACTED_FOLDER_NAME
        folder.mkdir(parents=True, exist_ok=True)
        for name, repo_path in MODEL_FILES.items():
            target = folder / name
            if target.exists():
                continue
            partial = target.with_suffix(target.suffix + ".part")
            with httpx.stream("GET", HF_MODEL_URL + repo_path, follow_redirects=True,
                              timeout=httpx.Timeout(30, read=120)) as response:
                response.raise_for_status()
                with open(partial, "wb") as f:
                    for data in response.iter_bytes(1 << 20):
                        f.write(data)
            partial.rename(target)  # only complete files count as downloaded


# --- Building the knowledge base -----------------------------------------------------------------
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


def pdf_files():
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    return sorted(DOCUMENTS_DIR.glob("*.pdf"))


def chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def build_knowledge_base(log=print):
    """
    Read every PDF in documents/, chunk and embed it, and store it in a fresh ChromaDB collection.
    Returns {"pdfs": n, "chunks": n}. Raises ValueError when there is nothing to index.
    """
    pdfs = pdf_files()
    if not pdfs:
        raise ValueError(f"No PDFs found. Put your domain PDF(s) in {DOCUMENTS_DIR} and run again.")

    ids, texts, metadatas = [], [], []
    for pdf in pdfs:
        try:
            pages = read_pdf(pdf)
        except Exception as e:
            log(f"  ✗ {pdf.name}: couldn't read it ({e}), skipping")
            continue
        if not pages:
            log(f"  ✗ {pdf.name}: no text found (scanned PDF?), skipping")
            continue
        count = 0
        for page, text in pages:
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{pdf.stem}-p{page}-c{i}")
                texts.append(chunk)
                metadatas.append({"source": pdf.name, "page": page})
                count += 1
        log(f"  ✓ {pdf.name}: {len(pages)} pages → {count} chunks")
    if not texts:
        raise ValueError("No text could be extracted from any PDF. Nothing to index.")

    if not MODEL_DIR.exists():
        log("Downloading the embedding model (all-MiniLM-L6-v2, ~80 MB, first time only)...")
    log(f"Embedding {len(texts)} chunks...")
    client = chroma_client()
    # Replace any previous index so removed or changed PDFs don't linger
    if COLLECTION_NAME in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME, embedding_function=ProjectMiniLM(), metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return {"pdfs": len({m["source"] for m in metadatas}), "chunks": collection.count()}


def load_collection():
    """Open the knowledge base built by build_knowledge_base()."""
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("Knowledge base not built yet. Run `python ingest.py`.")
    return chroma_client().get_collection(COLLECTION_NAME, embedding_function=ProjectMiniLM())


def indexed_pdfs(collection):
    return sorted({m["source"] for m in collection.get(include=["metadatas"])["metadatas"]})


# --- Question answering --------------------------------------------------------------------------
def retrieve(question, collection, k=TOP_K):
    """The k most similar chunks, as [{text, source, page, similarity}], most similar first."""
    k = min(k, collection.count())  # small knowledge bases may have fewer chunks than k
    result = collection.query(query_texts=[question], n_results=k, include=["documents", "metadatas", "distances"])
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


def answer(question, collection, stream_llm):
    """
    Returns (answer, sources). `answer` is NO_INFO when nothing relevant was found, otherwise a
    generator streaming the LLM's answer. `stream_llm(system_prompt, prompt)` yields text chunks.
    """
    sources = [c for c in retrieve(question, collection) if c["similarity"] >= MIN_SIMILARITY]
    if not sources:
        return NO_INFO, []
    return stream_llm(SYSTEM_PROMPT, build_prompt(question, sources)), sources


def is_no_info(reply):
    """The model was told to reply with NO_INFO when the context lacks the answer."""
    return NO_INFO.lower().rstrip(".") in reply.lower()


def ollama_ready():
    """True if Ollama is running and the model is pulled."""
    try:
        tags = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2).json()
    except (httpx.HTTPError, ValueError):
        return False
    return any(m["name"].split(":")[0] == OLLAMA_MODEL.split(":")[0] for m in tags.get("models", []))
