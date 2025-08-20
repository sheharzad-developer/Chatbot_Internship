import os
import json
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import asyncio

# Configure logging for serverless
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app optimized for serverless
app = FastAPI(
    title="AI Chatbot API",
    version="1.0.0",
    description="Lightweight AI Chatbot deployed on Vercel"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "vercel-user"
    session_id: Optional[str] = None
    brief_mode: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    metadata: dict = {}

# Simple lightweight AI service classes
class GroqService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        
    async def generate_content(self, prompt: str, model: str = "llama-3.3-70b-versatile", **kwargs) -> str:
        if not self.api_key:
            raise Exception("GROQ_API_KEY not set in environment variables")
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                raise Exception(f"Groq API error: {response.status_code} - {response.text}")

class GoogleAIService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_AI_GENERATIVE", "")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    async def generate_content(self, prompt: str, model: str = "gemini-1.5-flash", **kwargs) -> str:
        if not self.api_key:
            raise Exception("GOOGLE_AI_GENERATIVE not set in environment variables")
            
        url = f"{self.base_url}/models/{model}:generateContent"
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 800,
                "temperature": 0.3
            }
        }
        
        params = {"key": self.api_key}
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                params=params,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    return candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return "No response generated"
            else:
                raise Exception(f"Google AI API error: {response.status_code} - {response.text}")

# Initialize services
groq_service = GroqService()
google_ai_service = GoogleAIService()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "deployment": "vercel"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "deployment": "vercel",
        "services": {
            "groq": bool(os.getenv("GROQ_API_KEY")),
            "google_ai": bool(os.getenv("GOOGLE_AI_GENERATIVE"))
        }
    }

@app.post("/api/chat/groq", response_model=ChatResponse)
async def chat_groq(request: ChatRequest):
    """Chat endpoint using Groq API"""
    try:
        logger.info(f"Groq chat request: {request.message[:50]}...")
        
        # Create prompt
        system_prompt = "You are a helpful AI assistant. Provide clear, useful responses."
        if request.brief_mode:
            system_prompt += " Keep responses concise and direct."
        
        full_prompt = f"{system_prompt}\n\nUser: {request.message}\n\nAssistant:"
        
        # Generate response
        response = await groq_service.generate_content(full_prompt)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id or "vercel-session",
            metadata={
                "agent_type": "groq",
                "deployment": "vercel",
                "brief_mode": request.brief_mode
            }
        )
        
    except Exception as e:
        logger.error(f"Groq chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/google-ai", response_model=ChatResponse)
async def chat_google_ai(request: ChatRequest):
    """Chat endpoint using Google AI"""
    try:
        logger.info(f"Google AI chat request: {request.message[:50]}...")
        
        # Create prompt
        system_prompt = "You are a helpful AI assistant. Provide clear, useful responses."
        if request.brief_mode:
            system_prompt += " Keep responses concise and direct."
        
        full_prompt = f"{system_prompt}\n\nUser: {request.message}\n\nAssistant:"
        
        # Generate response
        response = await google_ai_service.generate_content(full_prompt)
        
        return ChatResponse(
            response=response,
            session_id=request.session_id or "vercel-session",
            metadata={
                "agent_type": "google_ai",
                "deployment": "vercel",
                "brief_mode": request.brief_mode
            }
        )
        
    except Exception as e:
        logger.error(f"Google AI chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
async def test_apis():
    """Test both AI APIs"""
    results = {}
    
    # Test Groq
    try:
        if os.getenv("GROQ_API_KEY"):
            test_response = await groq_service.generate_content("Say hello!")
            results["groq"] = {
                "status": "working",
                "response": test_response[:50] + "..." if len(test_response) > 50 else test_response
            }
        else:
            results["groq"] = {"status": "no_api_key"}
    except Exception as e:
        results["groq"] = {"status": "error", "error": str(e)}
    
    # Test Google AI
    try:
        if os.getenv("GOOGLE_AI_GENERATIVE"):
            test_response = await google_ai_service.generate_content("Say hello!")
            results["google_ai"] = {
                "status": "working", 
                "response": test_response[:50] + "..." if len(test_response) > 50 else test_response
            }
        else:
            results["google_ai"] = {"status": "no_api_key"}
    except Exception as e:
        results["google_ai"] = {"status": "error", "error": str(e)}
    
    return results

# For Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)