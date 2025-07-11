import logging
import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, documents_path: str = "documents", index_path: str = "vector_index"):
        self.documents_path = Path(documents_path)
        self.index_path = Path(index_path)
        self.documents_path.mkdir(exist_ok=True)
        self.index_path.mkdir(exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2
        
        # Initialize tokenizer for chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Document storage
        self.documents = []
        self.chunks = []
        self.chunk_metadata = []
        
        # Vector index
        self.index = None
        
        # Load existing index if available
        self._load_index()
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a document to the RAG system"""
        try:
            logger.info("Adding document to RAG system")
            
            # Chunk the document
            chunks = self._chunk_document(content)
            
            # Add to storage
            doc_id = len(self.documents)
            self.documents.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata or {}
            })
            
            # Add chunks with metadata
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "document_id": doc_id,
                    "chunk_id": i,
                    "document_metadata": metadata or {}
                }
                self.chunks.append(chunk)
                self.chunk_metadata.append(chunk_metadata)
            
            # Update vector index
            self._update_index()
            
            logger.info(f"Added document with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
    
    def search(self, query: str, top_k: int = 5) -> str:
        """Search for relevant documents"""
        try:
            if not self.chunks or self.index is None:
                return "No documents available for search."
            
            logger.info(f"Searching RAG system for: {query}")
            
            # Embed the query
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            # Search the index
            scores, indices = self.index.search(query_embedding, top_k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    metadata = self.chunk_metadata[idx]
                    
                    result = f"{i+1}. (Score: {score:.3f})\n{chunk}"
                    if metadata.get("document_metadata"):
                        result += f"\nMetadata: {metadata['document_metadata']}"
                    
                    results.append(result)
            
            if not results:
                return "No relevant documents found."
            
            formatted_results = "\n\n".join(results)
            logger.info(f"Found {len(results)} relevant chunks")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching RAG system: {e}")
            return f"Error searching documents: {str(e)}"
    
    def _chunk_document(self, content: str, max_tokens: int = 500, overlap: int = 50) -> List[str]:
        """Chunk document into smaller pieces"""
        try:
            # Tokenize the content
            tokens = self.tokenizer.encode(content)
            
            chunks = []
            start = 0
            
            while start < len(tokens):
                # Get chunk tokens
                end = min(start + max_tokens, len(tokens))
                chunk_tokens = tokens[start:end]
                
                # Decode back to text
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text.strip())
                
                # Move start position with overlap
                if end >= len(tokens):
                    break
                start = end - overlap
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return [content]  # Return original content if chunking fails
    
    def _update_index(self) -> None:
        """Update the FAISS vector index"""
        try:
            if not self.chunks:
                return
            
            logger.info("Updating vector index")
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_model.encode(self.chunks)
            embeddings = np.array(embeddings).astype('float32')
            
            # Create or update FAISS index
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
            else:
                self.index.reset()
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Save index
            self._save_index()
            
            logger.info(f"Vector index updated with {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error updating vector index: {e}")
    
    def _save_index(self) -> None:
        """Save the vector index and metadata"""
        try:
            # Save FAISS index
            index_file = self.index_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'chunks': self.chunks,
                    'chunk_metadata': self.chunk_metadata
                }, f)
            
            logger.info("Vector index and metadata saved")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _load_index(self) -> None:
        """Load existing vector index and metadata"""
        try:
            index_file = self.index_path / "faiss_index.bin"
            metadata_file = self.index_path / "metadata.pkl"
            
            if index_file.exists() and metadata_file.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_file))
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.chunks = data['chunks']
                    self.chunk_metadata = data['chunk_metadata']
                
                logger.info(f"Loaded existing index with {len(self.chunks)} chunks")
            else:
                logger.info("No existing index found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Starting with fresh index")
    
    def add_text_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a text file to the RAG system"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_metadata = {"source_file": file_path}
            if metadata:
                file_metadata.update(metadata)
            
            self.add_document(content, file_metadata)
            
        except Exception as e:
            logger.error(f"Error adding text file {file_path}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal if self.index else 0
        } 