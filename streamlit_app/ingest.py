"""
Build the Domain RAG knowledge base from the PDFs in documents/.

    python ingest.py

1. Reads every PDF in documents/ and splits each page into overlapping chunks.
2. Embeds the chunks with all-MiniLM-L6-v2 (downloaded once and saved to models/).
3. Stores them in a fresh ChromaDB collection in chroma_db/.

Run it again whenever you add, change or remove PDFs (or use "Rebuild" on the Domain RAG page).
"""
import sys

from rag import CHROMA_DIR, MODEL_DIR, build_knowledge_base


def main():
    try:
        stats = build_knowledge_base()
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"Saved embedding model in {MODEL_DIR.relative_to(MODEL_DIR.parents[1])}/")
    print(f"Done. {stats['chunks']} chunks from {stats['pdfs']} PDF(s) saved to {CHROMA_DIR.name}/")
    print("Now run:  streamlit run app.py   (then open 📚 Domain RAG)")


if __name__ == "__main__":
    main()
