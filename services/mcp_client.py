import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)

class MCPToolDefinition:
    """Represents a tool definition from MCP server"""
    def __init__(self, name: str, description: str, parameters: Dict, category: str = "general"):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.category = category
        self.prompt = ""

class MCPClient:
    """Client for communicating with MCP (Model Context Protocol) servers"""
    
    def __init__(self, tenant_id: str, server_config: Dict):
        self.tenant_id = tenant_id
        self.server_config = server_config
        self.server_url = server_config.get("url")
        self.auth_config = server_config.get("auth", {})
        self.connection_params = server_config.get("connection_params", {})
        
        # Connection state
        self.websocket = None
        self.is_connected = False
        self.connection_id = None
        
        # Tool registry
        self.available_tools: Dict[str, MCPToolDefinition] = {}
        self.tool_prompts: Dict[str, str] = {}
        
        # Rate limiting and security
        self.last_tool_calls: List[datetime] = []
        self.max_concurrent_calls = settings.max_concurrent_tool_calls
        self.active_calls = 0
        
        logger.info(f"Initialized MCP client for tenant {tenant_id} with server {self.server_url}")
    
    async def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            # Build connection headers if authentication is required
            headers = {}
            if self.auth_config.get("type") == "api_key":
                headers["Authorization"] = f"Bearer {self.auth_config.get('api_key')}"
            elif self.auth_config.get("type") == "bearer_token":
                headers["Authorization"] = f"Bearer {self.auth_config.get('token')}"
            
            # Connect to WebSocket
            timeout = self.connection_params.get("timeout", settings.mcp_server_timeout)
            
            self.websocket = await websockets.connect(
                self.server_url,
                extra_headers=headers,
                timeout=timeout,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_connected = True
            self.connection_id = str(uuid.uuid4())
            
            # Send initialization message
            await self._send_message({
                "jsonrpc": "2.0",
                "id": self.connection_id,
                "method": "initialize",
                "params": {
                    "client_info": {
                        "name": "Tool-Aware-Chatbot",
                        "version": settings.app_version
                    },
                    "tenant_id": self.tenant_id
                }
            })
            
            # Wait for initialization response
            response = await self._receive_message()
            if response.get("result", {}).get("success"):
                logger.info(f"Successfully connected to MCP server for tenant {self.tenant_id}")
                await self._discover_tools()
                return True
            else:
                logger.error(f"Failed to initialize MCP connection: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server for tenant {self.tenant_id}: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.websocket and self.is_connected:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.error(f"Error closing MCP connection: {e}")
            finally:
                self.is_connected = False
                self.websocket = None
                logger.info(f"Disconnected from MCP server for tenant {self.tenant_id}")
    
    async def _send_message(self, message: Dict):
        """Send a message to the MCP server"""
        if not self.websocket or not self.is_connected:
            raise ConnectionError("Not connected to MCP server")
        
        try:
            await self.websocket.send(json.dumps(message))
        except ConnectionClosed:
            self.is_connected = False
            raise ConnectionError("MCP server connection lost")
    
    async def _receive_message(self) -> Dict:
        """Receive a message from the MCP server"""
        if not self.websocket or not self.is_connected:
            raise ConnectionError("Not connected to MCP server")
        
        try:
            message = await self.websocket.recv()
            return json.loads(message)
        except ConnectionClosed:
            self.is_connected = False
            raise ConnectionError("MCP server connection lost")
    
    async def _discover_tools(self):
        """Discover available tools from the MCP server"""
        try:
            # Request tool list
            await self._send_message({
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "tools/list",
                "params": {}
            })
            
            response = await self._receive_message()
            tools_data = response.get("result", {}).get("tools", [])
            
            # Parse tool definitions
            for tool_data in tools_data:
                tool = MCPToolDefinition(
                    name=tool_data.get("name"),
                    description=tool_data.get("description", ""),
                    parameters=tool_data.get("inputSchema", {}),
                    category=tool_data.get("category", "general")
                )
                
                self.available_tools[tool.name] = tool
                logger.info(f"Discovered tool: {tool.name} for tenant {self.tenant_id}")
            
            logger.info(f"Discovered {len(self.available_tools)} tools for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
    
    def get_available_tools(self) -> List[Dict]:
        """Get list of available tools for the LLM"""
        tools_list = []
        for tool in self.available_tools.values():
            tools_list.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            })
        return tools_list
    
    def set_tool_prompts(self, tool_prompts: Dict[str, str]):
        """Set tool-specific prompts"""
        self.tool_prompts = tool_prompts
        for tool_name, prompt in tool_prompts.items():
            if tool_name in self.available_tools:
                self.available_tools[tool_name].prompt = prompt
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a tool on the MCP server"""
        if not self.is_connected:
            await self.connect()
        
        if tool_name not in self.available_tools:
            raise ValueError(f"Tool {tool_name} not available for tenant {self.tenant_id}")
        
        # Rate limiting check
        if self.active_calls >= self.max_concurrent_calls:
            raise RuntimeError("Maximum concurrent tool calls exceeded")
        
        # Security validation
        if not self._validate_tool_call(tool_name, parameters):
            raise SecurityError(f"Tool call validation failed for {tool_name}")
        
        self.active_calls += 1
        call_id = str(uuid.uuid4())
        
        try:
            # Execute tool
            await self._send_message({
                "jsonrpc": "2.0",
                "id": call_id,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            })
            
            # Wait for response with timeout
            response = await asyncio.wait_for(
                self._receive_message(),
                timeout=settings.mcp_tool_timeout
            )
            
            if "error" in response:
                raise RuntimeError(f"Tool execution error: {response['error']}")
            
            result = response.get("result", {})
            
            # Track tool call for rate limiting
            self.last_tool_calls.append(datetime.now())
            if len(self.last_tool_calls) > 100:  # Keep only last 100 calls
                self.last_tool_calls = self.last_tool_calls[-100:]
            
            logger.info(f"Successfully executed tool {tool_name} for tenant {self.tenant_id}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Tool {tool_name} execution timed out for tenant {self.tenant_id}")
            raise TimeoutError(f"Tool execution timed out")
        
        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed for tenant {self.tenant_id}: {e}")
            raise
        
        finally:
            self.active_calls -= 1
    
    def _validate_tool_call(self, tool_name: str, parameters: Dict) -> bool:
        """Validate tool call for security and rate limiting"""
        # Check if tool exists
        if tool_name not in self.available_tools:
            return False
        
        # Rate limiting check (last minute)
        now = datetime.now()
        recent_calls = [call for call in self.last_tool_calls if now - call < timedelta(minutes=1)]
        
        if len(recent_calls) > 60:  # Max 60 calls per minute per tenant
            logger.warning(f"Rate limit exceeded for tenant {self.tenant_id}")
            return False
        
        # Parameter validation (basic)
        tool = self.available_tools[tool_name]
        required_params = tool.parameters.get("required", [])
        
        for param in required_params:
            if param not in parameters:
                logger.warning(f"Missing required parameter {param} for tool {tool_name}")
                return False
        
        return True
    
    async def health_check(self) -> Dict:
        """Check the health of the MCP connection"""
        try:
            if not self.is_connected:
                return {
                    "status": "disconnected",
                    "server_url": self.server_url,
                    "tools_count": 0
                }
            
            # Send ping
            await self._send_message({
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "ping",
                "params": {}
            })
            
            response = await asyncio.wait_for(self._receive_message(), timeout=5)
            
            return {
                "status": "healthy",
                "server_url": self.server_url,
                "tools_count": len(self.available_tools),
                "active_calls": self.active_calls,
                "connection_id": self.connection_id
            }
            
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return {
                "status": "error",
                "server_url": self.server_url,
                "error": str(e)
            }

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass 