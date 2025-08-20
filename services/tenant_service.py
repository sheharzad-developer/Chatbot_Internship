import asyncio
import logging
from typing import Dict, Optional, List, Any
import yaml
from pathlib import Path
import redis.asyncio as redis
from cryptography.fernet import Fernet
import hashlib
import json
from datetime import datetime, timedelta

from config.settings import settings
from services.mcp_client import MCPClient
from models.tenant import TenantConfig, TenantSecuritySettings

logger = logging.getLogger(__name__)

class TenantService:
    """Service for managing multi-tenant configurations and security isolation"""
    
    def __init__(self):
        self.tenant_configs: Dict[str, TenantConfig] = {}
        self.mcp_clients: Dict[str, MCPClient] = {}
        self.redis_client = None
        self.encryption_key = None
        
        # Initialize encryption for sensitive data
        if settings.tenant_api_key_encryption_key:
            self.encryption_key = Fernet(settings.tenant_api_key_encryption_key.encode())
    
    async def initialize(self):
        """Initialize the tenant service"""
        try:
            # Try to connect to Redis for tenant isolation and caching
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    db=settings.redis_db_tenant_cache,
                    decode_responses=True
                )
                
                # Test Redis connection
                await self.redis_client.ping()
                logger.info("Connected to Redis for tenant caching")
            except Exception as redis_error:
                logger.warning(f"Redis connection failed: {redis_error}")
                logger.info("Continuing without Redis - using in-memory caching only")
                self.redis_client = None
            
            # Load all tenant configurations
            await self.load_all_tenant_configs()
            
            logger.info(f"Tenant service initialized with {len(self.tenant_configs)} tenants")
            
        except Exception as e:
            logger.error(f"Failed to initialize tenant service: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the tenant service and cleanup resources"""
        try:
            # Disconnect all MCP clients
            for tenant_id, mcp_client in self.mcp_clients.items():
                await mcp_client.disconnect()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Tenant service shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during tenant service shutdown: {e}")
    
    async def load_all_tenant_configs(self):
        """Load all tenant configurations from files"""
        config_dir = Path(settings.tenant_config_path)
        if not config_dir.exists():
            logger.warning(f"Tenant config directory not found: {config_dir}")
            return
        
        for config_file in config_dir.glob("*.yaml"):
            try:
                tenant_id = config_file.stem
                await self.load_tenant_config(tenant_id)
            except Exception as e:
                logger.error(f"Failed to load tenant config {config_file}: {e}")
    
    async def load_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Load a specific tenant configuration"""
        try:
            config_file = Path(settings.tenant_config_path) / f"{tenant_id}.yaml"
            
            if not config_file.exists():
                raise FileNotFoundError(f"Tenant config not found: {config_file}")
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Parse tenant configuration
            tenant_config = TenantConfig(**config_data)
            
            # Cache in memory and Redis
            self.tenant_configs[tenant_id] = tenant_config
            await self._cache_tenant_config(tenant_id, tenant_config)
            
            # Initialize MCP client if not already done
            if tenant_id not in self.mcp_clients:
                await self._initialize_mcp_client(tenant_id, tenant_config)
            
            logger.info(f"Loaded tenant config: {tenant_id}")
            return tenant_config
            
        except Exception as e:
            logger.error(f"Failed to load tenant config {tenant_id}: {e}")
            raise
    
    async def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration (from cache or load from file)"""
        # Try memory cache first
        if tenant_id in self.tenant_configs:
            return self.tenant_configs[tenant_id]
        
        # Try Redis cache
        cached_config = await self._get_cached_tenant_config(tenant_id)
        if cached_config:
            self.tenant_configs[tenant_id] = cached_config
            return cached_config
        
        # Load from file
        try:
            return await self.load_tenant_config(tenant_id)
        except Exception as e:
            logger.error(f"Failed to get tenant config {tenant_id}: {e}")
            return None
    
    async def get_mcp_client(self, tenant_id: str) -> Optional[MCPClient]:
        """Get MCP client for a tenant"""
        if tenant_id not in self.mcp_clients:
            tenant_config = await self.get_tenant_config(tenant_id)
            if tenant_config:
                await self._initialize_mcp_client(tenant_id, tenant_config)
        
        return self.mcp_clients.get(tenant_id)
    
    async def _initialize_mcp_client(self, tenant_id: str, tenant_config: TenantConfig):
        """Initialize MCP client for a tenant"""
        try:
            if not tenant_config.mcp_server or not tenant_config.tools.enabled:
                logger.info(f"MCP client not needed for tenant {tenant_id}")
                return
            
            # Create MCP client
            mcp_client = MCPClient(tenant_id, tenant_config.mcp_server)
            
            # Set tool prompts
            if tenant_config.tools.tool_prompts:
                mcp_client.set_tool_prompts(tenant_config.tools.tool_prompts)
            
            # Connect to MCP server
            if await mcp_client.connect():
                self.mcp_clients[tenant_id] = mcp_client
                logger.info(f"MCP client initialized for tenant {tenant_id}")
            else:
                logger.error(f"Failed to connect MCP client for tenant {tenant_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP client for tenant {tenant_id}: {e}")
    
    async def _cache_tenant_config(self, tenant_id: str, tenant_config: TenantConfig):
        """Cache tenant configuration in Redis"""
        try:
            if self.redis_client:
                config_json = tenant_config.json()
                await self.redis_client.setex(
                    f"tenant_config:{tenant_id}",
                    timedelta(hours=1),  # Cache for 1 hour
                    config_json
                )
        except Exception as e:
            logger.error(f"Failed to cache tenant config {tenant_id}: {e}")
    
    async def _get_cached_tenant_config(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get cached tenant configuration from Redis"""
        try:
            if self.redis_client:
                config_json = await self.redis_client.get(f"tenant_config:{tenant_id}")
                if config_json:
                    config_data = json.loads(config_json)
                    return TenantConfig(**config_data)
        except Exception as e:
            logger.error(f"Failed to get cached tenant config {tenant_id}: {e}")
        
        return None
    
    def validate_tenant_access(self, tenant_id: str, api_key: str = None) -> bool:
        """Validate tenant access and API key"""
        tenant_config = self.tenant_configs.get(tenant_id)
        if not tenant_config:
            return False
        
        # If no API key required, allow access
        if not api_key and not tenant_config.security.api_key_required:
            return True
        
        # Validate API key if provided
        if api_key and tenant_config.security.api_key_hash:
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            return api_key_hash == tenant_config.security.api_key_hash
        
        return False
    
    def get_tenant_database_prefix(self, tenant_id: str) -> str:
        """Get database collection prefix for tenant isolation"""
        if settings.tenant_isolation_enabled:
            return f"{tenant_id}_"
        return ""
    
    async def get_tenant_rate_limits(self, tenant_id: str) -> Dict:
        """Get rate limits for a tenant"""
        tenant_config = await self.get_tenant_config(tenant_id)
        if tenant_config and tenant_config.security.rate_limit:
            return {
                "requests_per_minute": tenant_config.security.rate_limit.requests_per_minute,
                "burst_limit": tenant_config.security.rate_limit.burst_limit
            }
        
        return {"requests_per_minute": 60, "burst_limit": 10}  # Default limits
    
    async def check_tenant_health(self, tenant_id: str) -> Dict:
        """Check health of tenant's MCP connection and services"""
        try:
            tenant_config = await self.get_tenant_config(tenant_id)
            if not tenant_config:
                return {"status": "tenant_not_found"}
            
            health_data = {
                "tenant_id": tenant_id,
                "tenant_name": tenant_config.tenant_name,
                "status": "healthy",
                "tools_enabled": tenant_config.tools.enabled,
                "mcp_server": None,
                "llm_provider": tenant_config.llm.provider if tenant_config.llm else "unknown"
            }
            
            # Check MCP client health
            mcp_client = await self.get_mcp_client(tenant_id)
            if mcp_client:
                health_data["mcp_server"] = await mcp_client.health_check()
            
            return health_data
            
        except Exception as e:
            logger.error(f"Health check failed for tenant {tenant_id}: {e}")
            return {
                "tenant_id": tenant_id,
                "status": "error",
                "error": str(e)
            }
    
    def list_tenants(self) -> List[str]:
        """List all available tenants"""
        return list(self.tenant_configs.keys())
    
    def get_tenant_info(self, tenant_id: str) -> Optional[Dict]:
        """Get basic tenant information"""
        tenant_config = self.tenant_configs.get(tenant_id)
        if tenant_config:
            return {
                "tenant_id": tenant_id,
                "tenant_name": tenant_config.tenant_name,
                "description": tenant_config.description,
                "tools_enabled": tenant_config.tools.enabled,
                "llm_provider": tenant_config.llm.provider if tenant_config.llm else "unknown",
                "tools_count": len(tenant_config.tools.allowed_categories) if tenant_config.tools else 0
            }
        return None

# Global tenant service instance
tenant_service = TenantService() 