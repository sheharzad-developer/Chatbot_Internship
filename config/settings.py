import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Dict, List, Optional
import yaml
from pathlib import Path

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    # MongoDB Configuration
    mongodb_url: str = os.getenv("MONGODB_URL", "")
    mongodb_db_name: str = "chatbot_db"
    
    # Redis Configuration (for tenant isolation and caching)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_db_tenant_cache: int = 0
    redis_db_session_cache: int = 1
    
    # Tavily API Configuration
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    
    # Langfuse Configuration (Enhanced)
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_host: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    langfuse_release: str = os.getenv("LANGFUSE_RELEASE", "production")
    
    # Google AI Configuration
    google_ai_generative: str = os.getenv("GOOGLE_AI_GENERATIVE", "")
    
    # DeepSeek API Configuration
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    # Groq API settings (FREE!)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    
    # XAI Configuration
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    
    # Multi-Tenant Configuration
    tenant_config_path: str = os.getenv("TENANT_CONFIG_PATH", "config/tenants")
    default_tenant: str = os.getenv("DEFAULT_TENANT", "default")
    tenant_isolation_enabled: bool = os.getenv("TENANT_ISOLATION_ENABLED", "true").lower() == "true"
    
    # MCP Server Configuration
    mcp_server_timeout: int = int(os.getenv("MCP_SERVER_TIMEOUT", "30"))
    mcp_tool_timeout: int = int(os.getenv("MCP_TOOL_TIMEOUT", "60"))
    mcp_max_retries: int = int(os.getenv("MCP_MAX_RETRIES", "3"))
    
    # Security Configuration
    tenant_api_key_encryption_key: str = os.getenv("TENANT_API_KEY_ENCRYPTION_KEY", "")
    max_tools_per_tenant: int = int(os.getenv("MAX_TOOLS_PER_TENANT", "50"))
    max_concurrent_tool_calls: int = int(os.getenv("MAX_CONCURRENT_TOOL_CALLS", "5"))
    
    # Application Configuration
    app_title: str = "Tool-Aware Multi-Tenant Chatbot"
    app_version: str = "2.0.0"
    debug: bool = True
    
    def get_tenant_config(self, tenant_id: str) -> Optional[Dict]:
        """Load tenant-specific configuration"""
        config_file = Path(self.tenant_config_path) / f"{tenant_id}.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return None
    
    def list_available_tenants(self) -> List[str]:
        """List all available tenant configurations"""
        config_dir = Path(self.tenant_config_path)
        if not config_dir.exists():
            return []
        
        tenants = []
        for config_file in config_dir.glob("*.yaml"):
            tenants.append(config_file.stem)
        return tenants
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields without validation errors

# Create global settings instance
settings = Settings() 