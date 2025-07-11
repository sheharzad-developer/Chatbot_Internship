#!/usr/bin/env python3
"""
Simple script to run the Reflect Agent Chatbot
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("🤖 Starting Reflect Agent Chatbot...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("💬 Chat Endpoint: http://localhost:8000/chat")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 