import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any

from config.settings import settings
from models.chat import (
    ChatRequest, ChatResponse, ChatHistoryResponse, MessageRole,
    ToolAwareChatRequest, ToolAwareChatResponse, TenantChatRequest
)
from models.tenant import TenantInfoResponse, TenantHealthResponse
from database.mongodb import db
from agents.simple_agent import SimpleAgent
from agents.gemini_agent import GeminiAgent
from agents.deepseek_agent import DeepSeekAgent
from agents.google_ai_agent import GoogleAIAgent
from agents.groq_agent import GroqAgent
from agents.tool_aware_agent import ToolAwareAgent
from services.rag_service import RAGService
from services.tenant_service import tenant_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances for backward compatibility
simple_agent = None
gemini_agent = None
deepseek_agent = None
google_ai_agent = None
groq_agent = None
rag_service = None

# New multi-tenant instances
tenant_agents: Dict[str, ToolAwareAgent] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global simple_agent, gemini_agent, deepseek_agent, google_ai_agent, groq_agent, rag_service
    
    # Startup
    logger.info("Starting up the Tool-Aware Multi-Tenant Chatbot...")
    
    try:
        # Connect to MongoDB
        await db.connect()
        
        # Initialize tenant service first
        await tenant_service.initialize()
        logger.info("✅ Tenant service initialized")
        
        # Initialize RAG service
        rag_service = RAGService()
        logger.info("✅ RAG service initialized")
        
        # Initialize backward-compatible agents
        simple_agent = SimpleAgent()
        logger.info("✅ Simple agent initialized")
        
        gemini_agent = GeminiAgent()
        logger.info("✅ Gemini agent initialized")
        
        deepseek_agent = DeepSeekAgent()
        logger.info("✅ DeepSeek agent initialized")
        
        google_ai_agent = GoogleAIAgent()
        logger.info("✅ Google AI agent initialized (FREE!)")
        
        groq_agent = GroqAgent()
        logger.info("✅ Groq agent initialized (FREE & FAST!)")
        
        # Initialize tool-aware agents for each tenant
        await initialize_tenant_agents()
        
        # Add sample document to RAG (optional)
        sample_doc = """
        This is a sample document for the RAG system. You can add your own documents
        through the API endpoints. The system supports document chunking and vector search
        using sentence transformers and FAISS.
        """
        rag_service.add_document(sample_doc, {"type": "sample", "title": "Sample Document"})
        
        logger.info("🚀 Tool-Aware Multi-Tenant Chatbot startup completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down the application...")
    try:
        # Shutdown tenant service
        await tenant_service.shutdown()
        
        # Disconnect from MongoDB
        await db.disconnect()
        
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

async def initialize_tenant_agents():
    """Initialize tool-aware agents for all tenants"""
    try:
        tenant_ids = tenant_service.list_tenants()
        
        for tenant_id in tenant_ids:
            try:
                agent = ToolAwareAgent(tenant_id)
                await agent.initialize()
                tenant_agents[tenant_id] = agent
                logger.info(f"✅ Tool-aware agent initialized for tenant: {tenant_id}")
            except Exception as e:
                logger.error(f"Failed to initialize agent for tenant {tenant_id}: {e}")
        
        logger.info(f"Initialized {len(tenant_agents)} tenant agents")
        
    except Exception as e:
        logger.error(f"Failed to initialize tenant agents: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description="A Tool-Aware Multi-Tenant Chatbot with MCP server integration, RAG, and comprehensive monitoring",
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

# Dependency functions for backward compatibility
def get_simple_agent() -> SimpleAgent:
    if simple_agent is None:
        raise HTTPException(status_code=500, detail="Simple agent not initialized")
    return simple_agent

def get_gemini_agent() -> GeminiAgent:
    if gemini_agent is None:
        raise HTTPException(status_code=500, detail="Gemini agent not initialized")
    return gemini_agent

def get_deepseek_agent() -> DeepSeekAgent:
    if deepseek_agent is None:
        raise HTTPException(status_code=500, detail="DeepSeek agent not initialized")
    return deepseek_agent

def get_google_ai_agent() -> GoogleAIAgent:
    if google_ai_agent is None:
        raise HTTPException(status_code=500, detail="Google AI agent not initialized")
    return google_ai_agent

def get_groq_agent() -> GroqAgent:
    if groq_agent is None:
        raise HTTPException(status_code=500, detail="Groq agent not initialized")
    return groq_agent

def get_rag_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    return rag_service

# New dependency functions for multi-tenant support
async def get_tenant_agent(tenant_id: str) -> ToolAwareAgent:
    """Get or create a tool-aware agent for a tenant"""
    if tenant_id not in tenant_agents:
        # Try to create agent for tenant
        try:
            agent = ToolAwareAgent(tenant_id)
            await agent.initialize()
            tenant_agents[tenant_id] = agent
            logger.info(f"Created new tool-aware agent for tenant: {tenant_id}")
        except Exception as e:
            logger.error(f"Failed to create agent for tenant {tenant_id}: {e}")
            raise HTTPException(status_code=404, detail=f"Tenant not found or invalid: {tenant_id}")
    
    return tenant_agents[tenant_id]

def validate_tenant_header(x_tenant_id: Optional[str] = Header(None)) -> str:
    """Validate tenant ID from header"""
    if not x_tenant_id:
        return settings.default_tenant
    
    # Validate tenant exists
    if not tenant_service.validate_tenant_access(x_tenant_id):
        raise HTTPException(status_code=403, detail="Invalid tenant ID")
    
    return x_tenant_id

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tool-Aware Multi-Tenant Chatbot API",
        "version": settings.app_version,
        "status": "running",
        "features": [
            "Multi-tenant isolation",
            "Tool-aware agents with MCP integration",
            "Comprehensive Langfuse tracking",
            "Dynamic tool discovery",
            "Secure tenant separation"
        ]
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    try:
        # Test MongoDB connection
        test_session = await db.get_all_chat_sessions(limit=1)
        
        # Test RAG service
        rag_stats = rag_service.get_stats() if rag_service else {}
        
        # Check tenant service health
        tenant_health = {}
        for tenant_id in tenant_service.list_tenants():
            tenant_health[tenant_id] = await tenant_service.check_tenant_health(tenant_id)
        
        return {
            "status": "healthy",
            "mongodb": "connected",
            "rag_system": rag_stats,
            "tenant_service": {
                "tenants_loaded": len(tenant_service.list_tenants()),
                "tenant_health": tenant_health
            },
            "legacy_services": {
                "simple_agent": simple_agent is not None,
                "gemini_agent": gemini_agent is not None,
                "deepseek_agent": deepseek_agent is not None,
                "google_ai_agent": google_ai_agent is not None,
                "groq_agent": groq_agent is not None,
                "rag_service": rag_service is not None
            },
            "tool_aware_agents": {
                "total_agents": len(tenant_agents),
                "active_tenants": list(tenant_agents.keys())
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

# ====================
# NEW MULTI-TENANT ENDPOINTS
# ====================

@app.get("/tenants")
async def list_tenants():
    """List all available tenants"""
    tenants = []
    for tenant_id in tenant_service.list_tenants():
        tenant_info = tenant_service.get_tenant_info(tenant_id)
        if tenant_info:
            tenants.append(tenant_info)
    
    return {"tenants": tenants, "total": len(tenants)}

@app.get("/tenants/{tenant_id}", response_model=TenantInfoResponse)
async def get_tenant_info(tenant_id: str):
    """Get detailed information about a specific tenant"""
    tenant_info = tenant_service.get_tenant_info(tenant_id)
    if not tenant_info:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return TenantInfoResponse(**tenant_info)

@app.get("/tenants/{tenant_id}/health", response_model=TenantHealthResponse)
async def get_tenant_health(tenant_id: str):
    """Get health status of a specific tenant"""
    health_data = await tenant_service.check_tenant_health(tenant_id)
    
    if health_data.get("status") == "tenant_not_found":
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return TenantHealthResponse(**health_data)

@app.post("/tenants/{tenant_id}/chat", response_model=ToolAwareChatResponse)
async def tenant_chat(
    tenant_id: str,
    request: TenantChatRequest
):
    """Main chat endpoint for tool-aware multi-tenant conversations"""
    try:
        # Get tenant agent
        agent = await get_tenant_agent(tenant_id)
        
        # Run the tool-aware agent
        response_text, tool_results, metadata = await agent.run(
            user_message=request.message,
            session_id=request.session_id,
            brief_mode=request.brief_mode,
            enable_tools=request.enable_tools
        )
        
        # Create or get session for database operations
        session_id = request.session_id
        if not session_id:
            # Get tenant database prefix for isolation
            db_prefix = tenant_service.get_tenant_database_prefix(tenant_id)
            session_id = await db.create_chat_session(
                user_id=request.user_id,
                title=f"Chat: {request.message[:50]}...",
                collection_prefix=db_prefix
            )
        
        # Add messages to session with tenant isolation
        db_prefix = tenant_service.get_tenant_database_prefix(tenant_id)
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message,
            collection_prefix=db_prefix
        )
        
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            collection_prefix=db_prefix,
            metadata={"tool_results": [result.dict() for result in tool_results]}
        )
        
        # Build comprehensive response
        return ToolAwareChatResponse(
            response=response_text,
            session_id=session_id,
            message_id=str(metadata.get("message_id", "")),
            tenant_id=tenant_id,
            tools_used=tool_results,
            tools_available=agent.get_available_tools(),
            response_time=metadata.get("execution_time", 0),
            trace_id=metadata.get("trace_id"),
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error in tenant chat for {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/tenants/{tenant_id}/tools/{tool_name}")
async def execute_tool(
    tenant_id: str,
    tool_name: str,
    parameters: Dict[str, Any]
):
    """Execute a specific tool for a tenant"""
    try:
        # Get MCP client for tenant
        mcp_client = await tenant_service.get_mcp_client(tenant_id)
        if not mcp_client:
            raise HTTPException(status_code=404, detail="No MCP client available for tenant")
        
        # Execute tool
        result = await mcp_client.execute_tool(tool_name, parameters)
        
        return {
            "tool_name": tool_name,
            "result": result,
            "tenant_id": tenant_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Tool execution failed for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Tool execution error: {str(e)}")

@app.get("/tenants/{tenant_id}/tools")
async def list_tenant_tools(tenant_id: str):
    """List available tools for a tenant"""
    try:
        agent = await get_tenant_agent(tenant_id)
        tools = agent.get_available_tools()
        
        return {
            "tenant_id": tenant_id,
            "tools": tools,
            "total": len(tools)
        }
        
    except Exception as e:
        logger.error(f"Failed to list tools for tenant {tenant_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tools: {str(e)}")

# ====================
# BACKWARD-COMPATIBLE ENDPOINTS
# ====================

@app.post("/chat/fast", response_model=ChatResponse)
async def fast_chat(
    request: ChatRequest,
    agent: SimpleAgent = Depends(get_simple_agent)
):
    """Ultra-fast chat endpoint - no database operations for maximum speed"""
    try:
        # Skip all database operations for speed
        # Get response from simple agent directly
        response = await agent.run(request.message)
        
        return ChatResponse(
            response=response,
            session_id="fast-mode",  # Static session ID
            metadata={"agent_type": "simple_agent", "mode": "fast"}
        )
        
    except Exception as e:
        logger.error(f"Error in fast chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Fast chat error: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    agent: SimpleAgent = Depends(get_simple_agent)
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
            metadata={"agent_type": "simple_agent"}
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/chat/gemini", response_model=ChatResponse)
async def chat_gemini(
    request: ChatRequest,
    agent: GeminiAgent = Depends(get_gemini_agent)
):
    """Chat endpoint using Google Gemini agent"""
    try:
        logger.info(f"Received Gemini chat request: {request.message}")
        
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = await db.create_chat_session(
                user_id=request.user_id,
                title=f"Gemini Chat: {request.message[:50]}..."
            )
        
        # Add user message to session
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        # Get response from Gemini agent
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
            metadata={"agent_type": "gemini_agent"}
        )
        
    except Exception as e:
        logger.error(f"Error in Gemini chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini chat error: {str(e)}")

@app.post("/chat/deepseek", response_model=ChatResponse)
async def chat_deepseek(
    request: ChatRequest,
    agent: DeepSeekAgent = Depends(get_deepseek_agent)
):
    """Chat endpoint using DeepSeek agent"""
    try:
        logger.info(f"Received DeepSeek chat request: {request.message}")
        
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = await db.create_chat_session(
                user_id=request.user_id,
                title=f"DeepSeek Chat: {request.message[:50]}..."
            )
        
        # Add user message to session
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        # Get response from DeepSeek agent - use brief mode if client prefers short responses
        if request.brief_mode:
            response = await agent.run_brief(request.message)
        else:
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
            metadata={"agent_type": "deepseek_agent", "brief_mode": request.brief_mode}
        )
        
    except Exception as e:
        logger.error(f"Error in DeepSeek chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"DeepSeek chat error: {str(e)}")

@app.post("/chat/google-ai", response_model=ChatResponse)
async def chat_google_ai(
    request: ChatRequest,
    agent: GoogleAIAgent = Depends(get_google_ai_agent)
):
    """Chat endpoint using FREE Google AI Studio"""
    try:
        logger.info(f"Received Google AI chat request: {request.message}")
        
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = await db.create_chat_session(
                user_id=request.user_id,
                title=f"Google AI Chat: {request.message[:50]}..."
            )
        
        # Add user message to session
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        # Get response from Google AI agent - use brief mode if requested
        if request.brief_mode:
            response = await agent.run_brief(request.message)
        else:
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
            metadata={"agent_type": "google_ai_agent", "brief_mode": request.brief_mode, "cost": "FREE"}
        )
        
    except Exception as e:
        logger.error(f"Error in Google AI chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Google AI chat error: {str(e)}")

@app.post("/chat/groq", response_model=ChatResponse)
async def chat_groq(
    request: ChatRequest,
    agent: GroqAgent = Depends(get_groq_agent)
):
    """Chat endpoint using FREE Groq API (Lightning Fast!)"""
    try:
        logger.info(f"Received Groq chat request: {request.message}")
        
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = await db.create_chat_session(
                user_id=request.user_id,
                title=f"Groq Chat: {request.message[:50]}..."
            )
        
        # Add user message to session
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        # Get response from Groq agent - use brief mode if requested
        if request.brief_mode:
            response = await agent.run_brief(request.message)
        else:
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
            metadata={"agent_type": "groq_agent", "brief_mode": request.brief_mode, "cost": "FREE"}
        )
        
    except Exception as e:
        logger.error(f"Error in Groq chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Groq chat error: {str(e)}")

@app.post("/chat/agent")
async def chat_with_agent_selection(
    request: ChatRequest,
    agent_type: str = Query(default="groq", description="Agent type: 'simple', 'gemini', 'deepseek', 'google-ai', or 'groq'")
):
    """Chat endpoint with agent selection"""
    try:
        if agent_type == "gemini":
            if gemini_agent is None:
                raise HTTPException(status_code=500, detail="Gemini agent not available")
            agent = gemini_agent
        elif agent_type == "deepseek":
            if deepseek_agent is None:
                raise HTTPException(status_code=500, detail="DeepSeek agent not available")
            agent = deepseek_agent
        elif agent_type == "google-ai":
            if google_ai_agent is None:
                raise HTTPException(status_code=500, detail="Google AI agent not available")
            agent = google_ai_agent
        elif agent_type == "groq":
            if groq_agent is None:
                raise HTTPException(status_code=500, detail="Groq agent not available")
            agent = groq_agent
        else:
            if simple_agent is None:
                raise HTTPException(status_code=500, detail="Simple agent not available") 
            agent = simple_agent
        
        logger.info(f"Received chat request for {agent_type} agent: {request.message}")
        
        # Create or get session
        session_id = request.session_id
        if not session_id:
            session_id = await db.create_chat_session(
                user_id=request.user_id,
                title=f"{agent_type.title()} Chat: {request.message[:50]}..."
            )
        
        # Add user message to session
        await db.add_message_to_session(
            session_id=session_id,
            role=MessageRole.USER,
            content=request.message
        )
        
        # Get response from selected agent - handle brief mode for DeepSeek, Google AI, and Groq
        if (agent_type == "deepseek" or agent_type == "google-ai" or agent_type == "groq") and request.brief_mode:
            response = await agent.run_brief(request.message)
        else:
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
            metadata={"agent_type": f"{agent_type}_agent", "brief_mode": request.brief_mode if agent_type in ["deepseek", "google-ai", "groq"] else False, "cost": "FREE" if agent_type in ["google-ai", "groq"] else "PAID"}
        )
        
    except Exception as e:
        logger.error(f"Error in agent selection chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/agents/test")
async def test_agents():
    """Test all available agents"""
    results = {}
    
    try:
        # Test simple agent
        if simple_agent:
            simple_response = await simple_agent.run("Hello, this is a test message")
            results["simple_agent"] = {
                "status": "working",
                "response": simple_response[:100] + "..." if len(simple_response) > 100 else simple_response
            }
        else:
            results["simple_agent"] = {"status": "not_available"}
        
        # Test Gemini agent
        if gemini_agent:
            gemini_test = gemini_agent.test_connection()
            if gemini_test:
                gemini_response = await gemini_agent.run("Hello, this is a test message")
                results["gemini_agent"] = {
                    "status": "working",
                    "response": gemini_response[:100] + "..." if len(gemini_response) > 100 else gemini_response
                }
            else:
                results["gemini_agent"] = {"status": "connection_failed"}
        else:
            results["gemini_agent"] = {"status": "not_available"}
        
        # Test DeepSeek agent
        if deepseek_agent:
            deepseek_test = deepseek_agent.test_connection()
            if deepseek_test:
                deepseek_response = await deepseek_agent.run("Hello, this is a test message")
                results["deepseek_agent"] = {
                    "status": "working",
                    "response": deepseek_response[:100] + "..." if len(deepseek_response) > 100 else deepseek_response
                }
            else:
                results["deepseek_agent"] = {"status": "connection_failed"}
        else:
            results["deepseek_agent"] = {"status": "not_available"}
    
    except Exception as e:
        logger.error(f"Error testing agents: {e}")
        results["error"] = str(e)
    
    return results

@app.get("/chat/history", response_model=ChatHistoryResponse)
async def get_chat_history(user_id: Optional[str] = None, limit: int = 50):
    """Get chat history"""
    try:
        logger.info(f"Getting chat history: user_id={user_id}, limit={limit}")
        if user_id:
            sessions = await db.get_user_chat_sessions(user_id, limit)
            logger.info(f"Retrieved {len(sessions)} sessions for user {user_id}")
        else:
            sessions = await db.get_all_chat_sessions(limit)
            logger.info(f"Retrieved {len(sessions)} total sessions")
        
        result = ChatHistoryResponse(
            sessions=sessions,
            total_count=len(sessions)
        )
        logger.info(f"Returning response with {len(result.sessions)} sessions")
        return result
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")

@app.get("/debug/db-test")
async def debug_db_test():
    """Debug endpoint to test database connection"""
    try:
        count = await db.chat_sessions_collection.count_documents({})
        sessions = await db.get_all_chat_sessions(2)
        return {
            "total_documents": count,
            "sessions_retrieved": len(sessions),
            "sample_session_ids": [s.session_id for s in sessions]
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}

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
    content: str = Form(..., description="Document content"),
    title: Optional[str] = Form(None, description="Document title"),
    rag: RAGService = Depends(get_rag_service)
):
    """Add a document to the RAG system"""
    try:
        doc_metadata = {}
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
    file: UploadFile = File(..., description="File to upload"),
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
    uvicorn.run(app, host="0.0.0.0", port=8001) 