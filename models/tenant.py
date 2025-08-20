from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from enum import Enum

class AuthType(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"

class LLMProvider(str, Enum):
    GROQ = "groq"
    GOOGLE_AI = "google-ai"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    SIMPLE = "simple"

class MCPServerAuth(BaseModel):
    type: AuthType = AuthType.NONE
    api_key: Optional[str] = None
    token: Optional[str] = None

class MCPServerConnectionParams(BaseModel):
    timeout: int = 30
    max_retries: int = 3

class MCPServerConfig(BaseModel):
    url: str
    protocol: str = "websocket"
    auth: MCPServerAuth = MCPServerAuth()
    connection_params: MCPServerConnectionParams = MCPServerConnectionParams()

class ToolsConfig(BaseModel):
    enabled: bool = True
    max_concurrent: int = 5
    timeout: int = 60
    allowed_categories: List[str] = []
    tool_prompts: Optional[Dict[str, str]] = {}

class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.GROQ
    model: str = "llama3-8b-8192"
    temperature: float = 0.7
    max_tokens: int = 1000
    brief_mode_max_tokens: int = 200

class RateLimit(BaseModel):
    requests_per_minute: int = 60
    burst_limit: int = 10

class ContentFiltering(BaseModel):
    enabled: bool = True
    block_personal_info: bool = True
    block_sensitive_data: bool = True
    custom_filters: Optional[List[str]] = []

class TenantSecuritySettings(BaseModel):
    rate_limit: RateLimit = RateLimit()
    content_filtering: ContentFiltering = ContentFiltering()
    api_key_required: bool = False
    api_key_hash: Optional[str] = None

class LangfuseConfig(BaseModel):
    enabled: bool = True
    session_tracking: bool = True
    tool_call_tracking: bool = True
    cost_tracking: bool = True
    detailed_tracing: bool = False

class MonitoringConfig(BaseModel):
    langfuse: LangfuseConfig = LangfuseConfig()
    custom_tags: List[str] = []

class DatabaseConfig(BaseModel):
    isolation: bool = True
    collection_prefix: str = ""
    retention_days: int = 90
    backup_enabled: bool = False
    encryption_at_rest: bool = False

class TenantConfig(BaseModel):
    tenant_id: str
    tenant_name: str
    description: str = ""
    
    # MCP Server Configuration
    mcp_server: Optional[MCPServerConfig] = None
    
    # System Prompt
    system_prompt: str = "You are a helpful AI assistant."
    
    # Tool Configuration
    tools: ToolsConfig = ToolsConfig()
    
    # LLM Configuration
    llm: LLMConfig = LLMConfig()
    
    # Security Settings
    security: TenantSecuritySettings = TenantSecuritySettings()
    
    # Monitoring and Logging
    monitoring: MonitoringConfig = MonitoringConfig()
    
    # Database Configuration
    database: DatabaseConfig = DatabaseConfig()
    
    class Config:
        use_enum_values = True

class TenantCreateRequest(BaseModel):
    tenant_name: str
    description: Optional[str] = ""
    
    # Optional configurations (will use defaults if not provided)
    mcp_server: Optional[MCPServerConfig] = None
    system_prompt: Optional[str] = None
    tools: Optional[ToolsConfig] = None
    llm: Optional[LLMConfig] = None
    security: Optional[TenantSecuritySettings] = None
    monitoring: Optional[MonitoringConfig] = None
    database: Optional[DatabaseConfig] = None

class TenantUpdateRequest(BaseModel):
    tenant_name: Optional[str] = None
    description: Optional[str] = None
    mcp_server: Optional[MCPServerConfig] = None
    system_prompt: Optional[str] = None
    tools: Optional[ToolsConfig] = None
    llm: Optional[LLMConfig] = None
    security: Optional[TenantSecuritySettings] = None
    monitoring: Optional[MonitoringConfig] = None
    database: Optional[DatabaseConfig] = None

class TenantInfoResponse(BaseModel):
    tenant_id: str
    tenant_name: str
    description: str
    tools_enabled: bool
    llm_provider: str
    tools_count: int
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class TenantHealthResponse(BaseModel):
    tenant_id: str
    tenant_name: str
    status: str
    tools_enabled: bool
    mcp_server: Optional[Dict[str, Any]] = None
    llm_provider: str
    error: Optional[str] = None

class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    
class ToolCallResponse(BaseModel):
    tool_name: str
    result: Dict[str, Any]
    execution_time: float
    status: str
    error: Optional[str] = None 