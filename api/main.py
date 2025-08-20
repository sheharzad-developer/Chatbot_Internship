import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import our modules
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

# Import your existing modules
from config.settings import settings
from models.chat import ChatRequest, ChatResponse
from agents.groq_agent import GroqAgent
from agents.google_ai_agent import GoogleAIAgent

# Configure logging for serverless
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app optimized for serverless
app = FastAPI(
    title="Chatbot API",
    version="1.0.0",
    description="AI Chatbot deployed on Vercel"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize lightweight agents (no MongoDB for now)
groq_agent = GroqAgent()
google_ai_agent = GoogleAIAgent()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "deployment": "vercel",
        "agents": {
            "groq_agent": groq_agent is not None,
            "google_ai_agent": google_ai_agent is not None
        }
    }

@app.post("/api/chat/groq", response_model=ChatResponse)
async def chat_groq(request: ChatRequest):
    """Chat endpoint using FREE Groq API"""
    try:
        logger.info(f"Received Groq chat request: {request.message}")
        
        # Get response from Groq agent
        if request.brief_mode:
            response = await groq_agent.run_brief(request.message)
        else:
            response = await groq_agent.run(request.message, request.session_id)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id or "vercel-session",
            metadata={"agent_type": "groq_agent", "deployment": "vercel"}
        )
        
    except Exception as e:
        logger.error(f"Error in Groq chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/chat/google-ai", response_model=ChatResponse)
async def chat_google_ai(request: ChatRequest):
    """Chat endpoint using Google AI Studio"""
    try:
        logger.info(f"Received Google AI chat request: {request.message}")
        
        # Get response from Google AI agent
        if request.brief_mode:
            response = await google_ai_agent.run_brief(request.message)
        else:
            response = await google_ai_agent.run(request.message, request.session_id)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id or "vercel-session",
            metadata={"agent_type": "google_ai_agent", "deployment": "vercel"}
        )
        
    except Exception as e:
        logger.error(f"Error in Google AI chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/api/agents/test")
async def test_agents():
    """Test available agents"""
    results = {}
    
    try:
        # Test Groq agent
        if groq_agent:
            test_response = await groq_agent.run("Hello, this is a test message")
            results["groq_agent"] = {
                "status": "working",
                "response": test_response[:100] + "..." if len(test_response) > 100 else test_response
            }
        
        # Test Google AI agent  
        if google_ai_agent:
            test_response = await google_ai_agent.run("Hello, this is a test message")
            results["google_ai_agent"] = {
                "status": "working", 
                "response": test_response[:100] + "..." if len(test_response) > 100 else test_response
            }
            
    except Exception as e:
        logger.error(f"Error testing agents: {e}")
        results["error"] = str(e)
    
    return results

# This is important for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
