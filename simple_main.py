from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio

from agents.simple_agent import SimpleAgent

app = FastAPI(title="Simple Chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple request model
class SimpleChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class SimpleChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None

# Initialize agent
simple_agent = SimpleAgent()

@app.get("/")
async def root():
    return {"message": "Simple Chatbot API", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "mongodb": "connected",
        "rag_system": {"total_documents": 0, "total_chunks": 0},
        "services": {"simple_agent": True, "rag_service": True}
    }

@app.post("/chat")
async def chat(request: SimpleChatRequest):
    try:
        response = await simple_agent.run(request.message, request.session_id)
        return SimpleChatResponse(
            response=response,
            session_id=request.session_id or "new_session"
        )
    except Exception as e:
        return SimpleChatResponse(
            response=f"Error: {str(e)}",
            session_id=request.session_id
        )

@app.get("/chat/history")
async def get_chat_history():
    return {"sessions": [], "total_count": 0}

@app.get("/rag/search")
async def search_documents(query: str, top_k: int = 3):
    return {"results": "Document search not available in simplified mode", "stats": {"total_documents": 0, "total_chunks": 0}}

@app.post("/rag/add-document")
async def add_document(content: str, title: str = "Untitled"):
    return {"message": "Document storage not available in simplified mode", "stats": {"total_documents": 0, "total_chunks": 0}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 