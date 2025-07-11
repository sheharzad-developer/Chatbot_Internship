import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from config.settings import settings
from models.chat import ChatRequest, ChatResponse, ChatHistoryResponse, MessageRole
from database.mongodb import db
from agents.reflect_agent import ReflectAgent
from services.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
reflect_agent = None
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global reflect_agent, rag_service
    
    # Startup
    logger.info("Starting up the application...")
    
    try:
        # Connect to MongoDB
        await db.connect()
        
        # Initialize services
        reflect_agent = ReflectAgent()
        rag_service = RAGService()
        
        # Add sample document to RAG (optional)
        sample_doc = """
        This is a sample document for the RAG system. You can add your own documents
        through the API endpoints. The system supports document chunking and vector search
        using sentence transformers and FAISS.
        """
        rag_service.add_document(sample_doc, {"type": "sample", "title": "Sample Document"})
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down the application...")
    try:
        await db.disconnect()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="A Reflect Agent Chatbot with RAG, Tavily search, and MongoDB chat history",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get services
def get_reflect_agent() -> ReflectAgent:
    if reflect_agent is None:
        raise HTTPException(status_code=500, detail="Reflect agent not initialized")
    return reflect_agent

def get_rag_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    return rag_service

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Reflect Agent Chatbot API",
        "version": settings.app_version,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        test_session = await db.get_all_chat_sessions(limit=1)
        
        # Test RAG service
        rag_stats = rag_service.get_stats() if rag_service else {}
        
        return {
            "status": "healthy",
            "mongodb": "connected",
            "rag_system": rag_stats,
            "services": {
                "reflect_agent": reflect_agent is not None,
                "rag_service": rag_service is not None
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: ReflectAgent = Depends(get_reflect_agent)
):
    """Main chat endpoint"""
    try:
        logger.info(f"Received chat request: {request.message}")
        
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = await db.create_chat_session(
                user_id=request.user_id,
                title=f"Chat about: {request.message[:50]}..."
            )
        
        # Add user message to session
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        # Get response from reflect agent
        response = await agent.run(request.message, session_id)
        
        # Add assistant response to session
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=response
        )
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            metadata={"agent_type": "reflect_agent"}
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(user_id: Optional[str] = None, limit: int = 50):
    """Get chat history"""
    try:
        if user_id:
            sessions = await db.get_user_chat_sessions(user_id, limit)
        else:
            sessions = await db.get_all_chat_sessions(limit)
        
        return ChatHistoryResponse(
            sessions=sessions,
            total_count=len(sessions)
        )
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")

@app.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str):
    """Get a specific chat session"""
    try:
        session = await db.get_chat_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat session: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")

@app.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    try:
        success = await db.delete_chat_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")

@app.post("/rag/add-document")
async def add_document(
    content: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
    rag: RAGService = Depends(get_rag_service)
):
    """Add a document to the RAG system"""
    try:
        doc_metadata = metadata or {}
        if title:
            doc_metadata["title"] = title
        
        rag.add_document(content, doc_metadata)
        
        return {
            "message": "Document added successfully",
            "stats": rag.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@app.post("/rag/upload-file")
async def upload_file(
    file: UploadFile = File(...),
    rag: RAGService = Depends(get_rag_service)
):
    """Upload and add a text file to the RAG system"""
    try:
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Add to RAG system
        metadata = {
            "filename": file.filename,
            "content_type": file.content_type,
            "uploaded": True
        }
        
        rag.add_document(text_content, metadata)
        
        return {
            "message": f"File '{file.filename}' uploaded and added successfully",
            "stats": rag.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/rag/search")
async def search_documents(
    query: str,
    top_k: int = 5,
    rag: RAGService = Depends(get_rag_service)
):
    """Search documents in the RAG system"""
    try:
        results = rag.search(query, top_k)
        
        return {
            "query": query,
            "results": results,
            "stats": rag.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")

@app.get("/rag/stats")
async def get_rag_stats(rag: RAGService = Depends(get_rag_service)):
    """Get RAG system statistics"""
    try:
        return rag.get_stats()
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting RAG stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 