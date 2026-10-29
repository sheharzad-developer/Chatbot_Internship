"""
Build the vector store from the PDFs in documents/.

    python ingest.py

1. Downloads the embedding model once and saves it to models/ (later runs reuse it, offline).
2. Reads every PDF in documents/ and splits each page into overlapping chunks.
3. Embeds the chunks and stores them in a fresh ChromaDB collection in chroma_db/.

Run it again whenever you add, change or remove PDFs.
"""
import sys

from sentence_transformers import SentenceTransformer

from rag import (
    CHROMA_DIR, COLLECTION_NAME, DOCUMENTS_DIR, EMBEDDING_MODEL_NAME, MODEL_DIR,
    chroma_client, chunk_text, read_pdf,
)


def main():
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found. Put your domain PDF(s) in {DOCUMENTS_DIR} and run again.")
        sys.exit(1)

    # 1. Embedding model: download once, then load from disk
    if MODEL_DIR.exists():
        print(f"Loading embedding model from {MODEL_DIR.relative_to(MODEL_DIR.parents[1])}")
        embedder = SentenceTransformer(str(MODEL_DIR))
    else:
        print(f"Downloading embedding model {EMBEDDING_MODEL_NAME}...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embedder.save(str(MODEL_DIR))
        print(f"Saved model to {MODEL_DIR.relative_to(MODEL_DIR.parents[1])}")

    # 2. Read and chunk the PDFs
    ids, texts, metadatas = [], [], []
    for pdf in pdfs:
        try:
            pages = read_pdf(pdf)
        except Exception as e:
            print(f"  ✗ {pdf.name}: couldn't read it ({e}), skipping")
            continue
        if not pages:
            print(f"  ✗ {pdf.name}: no text found (scanned PDF?), skipping")
            continue
        count = 0
        for page, text in pages:
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{pdf.stem}-p{page}-c{i}")
                texts.append(chunk)
                metadatas.append({"source": pdf.name, "page": page})
                count += 1
        print(f"  ✓ {pdf.name}: {len(pages)} pages → {count} chunks")

    if not texts:
        print("No text could be extracted from any PDF. Nothing to index.")
        sys.exit(1)

    # 3. Embed and store, replacing any previous index so removed PDFs don't linger
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    client = chroma_client()
    if COLLECTION_NAME in [c if isinstance(c, str) else c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings.tolist())

    print(f"Done. {collection.count()} chunks from {len(set(m['source'] for m in metadatas))} PDF(s) "
          f"saved to {CHROMA_DIR.name}/")
    print("Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
