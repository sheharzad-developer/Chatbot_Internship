from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"

class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    parameters: Dict[str, Any]

class ToolCallResult(BaseModel):
    tool_call_id: str
    name: str
    result: Any
    execution_time: float
    status: str = "success"  # success, error, timeout
    error: Optional[str] = None

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Tool calling support
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_results: Optional[List[ToolCallResult]] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Fixed default value

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = "default"  # Default value for backward compatibility
    user_id: Optional[str] = None
    title: str
    messages: List[ChatMessage] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Session metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    message: str
    tenant_id: str = Field(default="default", description="Tenant identifier")
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    brief_mode: bool = False
    
    # Tool calling options
    enable_tools: bool = True
    allowed_tools: Optional[List[str]] = None  # Restrict to specific tools
    
    # Additional context
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: Optional[str] = None
    
    # Tool calling information
    tool_calls_made: Optional[List[ToolCallResult]] = []
    tools_available: Optional[List[str]] = []
    
    # Response metadata
    metadata: Optional[Dict[str, Any]] = {}

class ChatHistoryResponse(BaseModel):
    sessions: List[ChatSession]
    total_count: int
    tenant_id: Optional[str] = None

class ToolAwareChatRequest(BaseModel):
    """Enhanced chat request with full tool-aware capabilities"""
    message: str
    tenant_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    
    # Tool configuration
    enable_tools: bool = True
    allowed_tools: Optional[List[str]] = None
    max_tool_calls: int = 5
    tool_timeout: int = 60
    
    # Response configuration
    brief_mode: bool = False
    include_tool_details: bool = False
    stream_response: bool = False
    
    # Context and metadata
    context: Optional[Dict[str, Any]] = {}
    user_preferences: Optional[Dict[str, Any]] = {}

class ToolAwareChatResponse(BaseModel):
    """Enhanced chat response with comprehensive tool information"""
    response: str
    session_id: str
    message_id: str
    tenant_id: str
    
    # Tool execution details
    tools_used: List[ToolCallResult] = []
    tools_available: List[str] = []
    tool_execution_summary: Optional[Dict[str, Any]] = {}
    
    # Response metadata
    response_time: float
    token_usage: Optional[Dict[str, int]] = {}
    cost_info: Optional[Dict[str, Any]] = {}
    
    # Langfuse tracking
    trace_id: Optional[str] = None
    observation_id: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = {}

class TenantChatRequest(BaseModel):
    """Simplified tenant-aware chat request"""
    message: str
    brief_mode: bool = False
    enable_tools: bool = True
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous" 