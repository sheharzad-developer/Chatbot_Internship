import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import uuid
import time

from langfuse import Langfuse
try:
    from langfuse.decorators import observe
except ImportError:
    # Fallback for older versions
    def observe(func):
        return func

from config.settings import settings
from models.chat import ToolCall, ToolCallResult, ChatMessage, MessageRole
from models.tenant import TenantConfig
from services.mcp_client import MCPClient
from services.tenant_service import tenant_service

# Import LLM providers
from services.groq_service import GroqService
from services.google_ai_service import GoogleAIService
from services.deepseek_service import DeepSeekService
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class ToolAwareAgent:
    """Advanced AI agent with tool-calling capabilities and multi-tenant support"""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.tenant_config: Optional[TenantConfig] = None
        self.mcp_client: Optional[MCPClient] = None
        self.llm_service = None
        
        # Langfuse for observability
        self.langfuse = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host,
            release=settings.langfuse_release
        )
        
        # Tool execution tracking
        self.tool_call_history: List[ToolCallResult] = []
        self.max_tool_calls_per_conversation = 10
        
        logger.info(f"Initialized ToolAwareAgent for tenant: {tenant_id}")
    
    async def initialize(self):
        """Initialize the agent with tenant configuration"""
        try:
            # Load tenant configuration
            self.tenant_config = await tenant_service.get_tenant_config(self.tenant_id)
            if not self.tenant_config:
                raise ValueError(f"Tenant configuration not found: {self.tenant_id}")
            
            # Initialize MCP client if tools are enabled
            if self.tenant_config.tools.enabled:
                self.mcp_client = await tenant_service.get_mcp_client(self.tenant_id)
            
            # Initialize LLM service based on tenant configuration
            await self._initialize_llm_service()
            
            logger.info(f"ToolAwareAgent initialized for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ToolAwareAgent for tenant {self.tenant_id}: {e}")
            raise
    
    async def _initialize_llm_service(self):
        """Initialize the appropriate LLM service based on tenant configuration"""
        if not self.tenant_config or not self.tenant_config.llm:
            logger.warning(f"No LLM configuration found for tenant {self.tenant_id}")
            return
        
        provider = self.tenant_config.llm.provider
        
        try:
            if provider == "groq":
                self.llm_service = GroqService()
            elif provider == "google-ai":
                self.llm_service = GoogleAIService()
            elif provider == "deepseek":
                self.llm_service = DeepSeekService()
            elif provider == "gemini":
                self.llm_service = GeminiService()
            else:
                logger.warning(f"Unknown LLM provider {provider} for tenant {self.tenant_id}")
                # Default to Groq
                self.llm_service = GroqService()
            
            logger.info(f"Initialized {provider} LLM service for tenant {self.tenant_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service for tenant {self.tenant_id}: {e}")
            raise
    
    @observe()
    async def run(
        self,
        user_message: str,
        session_id: Optional[str] = None,
        brief_mode: bool = False,
        enable_tools: bool = True,
        allowed_tools: Optional[List[str]] = None,
        max_tool_calls: int = 5,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, List[ToolCallResult], Dict[str, Any]]:
        """
        Run the tool-aware agent with comprehensive tool calling capabilities
        
        Returns:
            Tuple of (response_text, tool_results, metadata)
        """
        start_time = time.time()
        tool_results: List[ToolCallResult] = []
        conversation_context = context or {}
        
        try:
            # Create Langfuse trace
            trace = self.langfuse.trace(
                name=f"tool_aware_chat_{self.tenant_id}",
                session_id=session_id,
                user_id=conversation_context.get("user_id", "anonymous"),
                metadata={
                    "tenant_id": self.tenant_id,
                    "brief_mode": brief_mode,
                    "enable_tools": enable_tools,
                    "max_tool_calls": max_tool_calls
                }
            )
            
            # Build system prompt
            system_prompt = self._build_system_prompt(brief_mode, enable_tools)
            
            # Get available tools
            available_tools = []
            if enable_tools and self.mcp_client and self.tenant_config.tools.enabled:
                available_tools = self._get_filtered_tools(allowed_tools)
            
            # Build messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Add tool information to messages if tools are available
            if available_tools:
                tool_info = self._format_tools_for_llm(available_tools)
                messages[0]["content"] += f"\n\nAvailable tools:\n{tool_info}"
            
            # Track LLM generation
            llm_generation = trace.generation(
                name=f"llm_generation_{self.tenant_config.llm.provider}",
                model=self.tenant_config.llm.model,
                input=messages,
                metadata={
                    "temperature": self.tenant_config.llm.temperature,
                    "max_tokens": self.tenant_config.llm.brief_mode_max_tokens if brief_mode else self.tenant_config.llm.max_tokens
                }
            )
            
            # Generate initial response
            response_text = await self._generate_llm_response(
                messages=messages,
                brief_mode=brief_mode,
                available_tools=available_tools
            )
            
            # Process tool calls if any
            if enable_tools and available_tools:
                response_text, tool_results = await self._process_tool_calls(
                    response_text=response_text,
                    available_tools=available_tools,
                    max_tool_calls=max_tool_calls,
                    trace=trace,
                    conversation_context=conversation_context
                )
            
            # Finalize LLM generation tracking
            llm_generation.end(
                output=response_text,
                metadata={
                    "tool_calls_made": len(tool_results),
                    "tools_used": [result.name for result in tool_results]
                }
            )
            
            # Calculate response metadata
            execution_time = time.time() - start_time
            metadata = {
                "tenant_id": self.tenant_id,
                "execution_time": execution_time,
                "tools_available": len(available_tools),
                "tools_used": len(tool_results),
                "llm_provider": self.tenant_config.llm.provider,
                "brief_mode": brief_mode,
                "trace_id": trace.id
            }
            
            # End trace
            trace.update(
                output=response_text,
                metadata=metadata
            )
            
            logger.info(f"ToolAwareAgent completed for tenant {self.tenant_id} in {execution_time:.2f}s")
            
            return response_text, tool_results, metadata
            
        except Exception as e:
            logger.error(f"Error in ToolAwareAgent for tenant {self.tenant_id}: {e}")
            
            # Track error in Langfuse
            if 'trace' in locals():
                trace.update(
                    metadata={"error": str(e)},
                    tags=["error"]
                )
            
            # Return error response
            error_response = "I apologize, but I encountered an error while processing your request. Please try again."
            if brief_mode:
                error_response = "Error processing request. Please try again."
            
            return error_response, [], {"error": str(e), "tenant_id": self.tenant_id}
    
    def _build_system_prompt(self, brief_mode: bool, enable_tools: bool) -> str:
        """Build the system prompt based on tenant configuration and settings"""
        base_prompt = self.tenant_config.system_prompt
        
        if brief_mode:
            base_prompt += "\n\nIMPORTANT: Provide very brief, concise responses. Keep answers short and to the point."
        
        if enable_tools and self.tenant_config.tools.enabled:
            base_prompt += "\n\nYou have access to tools that can help you provide accurate information. When appropriate, use these tools to enhance your responses. Always explain what tools you're using and why."
            
            # Add tool-specific prompts
            if self.tenant_config.tools.tool_prompts:
                base_prompt += "\n\nTool-specific guidance:"
                for tool_name, prompt in self.tenant_config.tools.tool_prompts.items():
                    base_prompt += f"\n- {tool_name}: {prompt}"
        
        return base_prompt
    
    def _get_filtered_tools(self, allowed_tools: Optional[List[str]]) -> List[Dict]:
        """Get filtered list of available tools"""
        if not self.mcp_client:
            return []
        
        all_tools = self.mcp_client.get_available_tools()
        
        if not allowed_tools:
            return all_tools
        
        # Filter tools based on allowed list
        filtered_tools = []
        for tool in all_tools:
            if tool["function"]["name"] in allowed_tools:
                filtered_tools.append(tool)
        
        return filtered_tools
    
    def _format_tools_for_llm(self, available_tools: List[Dict]) -> str:
        """Format available tools for LLM consumption"""
        if not available_tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in available_tools:
            func = tool["function"]
            tool_descriptions.append(f"- {func['name']}: {func['description']}")
        
        return "\n".join(tool_descriptions)
    
    async def _generate_llm_response(
        self,
        messages: List[Dict],
        brief_mode: bool,
        available_tools: List[Dict]
    ) -> str:
        """Generate response from LLM"""
        if not self.llm_service:
            raise RuntimeError("LLM service not initialized")
        
        max_tokens = (
            self.tenant_config.llm.brief_mode_max_tokens if brief_mode 
            else self.tenant_config.llm.max_tokens
        )
        
        # Convert messages to prompt format for most services
        prompt = self._convert_messages_to_prompt(messages)
        
        # Prepare LLM request
        request_params = {
            "prompt": prompt,
            "temperature": self.tenant_config.llm.temperature,
            "max_tokens": max_tokens
        }
        
        # Add tools if available and supported
        if available_tools and hasattr(self.llm_service, 'supports_function_calling'):
            if self.llm_service.supports_function_calling():
                request_params["tools"] = available_tools
        
        # Generate response
        response = await self.llm_service.generate_content(**request_params)
        
        return response
    
    def _convert_messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages format to a single prompt string"""
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(content)
        
        # Add a final prompt for the assistant to respond
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def _process_tool_calls(
        self,
        response_text: str,
        available_tools: List[Dict],
        max_tool_calls: int,
        trace,
        conversation_context: Dict[str, Any]
    ) -> Tuple[str, List[ToolCallResult]]:
        """Process any tool calls in the response"""
        tool_results: List[ToolCallResult] = []
        
        # Parse tool calls from response (this is simplified - in practice, you'd need
        # more sophisticated parsing based on the LLM's response format)
        tool_calls = self._extract_tool_calls_from_response(response_text)
        
        if not tool_calls or len(tool_calls) == 0:
            return response_text, tool_results
        
        # Execute tool calls
        for i, tool_call in enumerate(tool_calls[:max_tool_calls]):
            try:
                # Track tool execution in Langfuse
                tool_span = trace.span(
                    name=f"tool_execution_{tool_call.name}",
                    input=tool_call.parameters,
                    metadata={
                        "tool_name": tool_call.name,
                        "tenant_id": self.tenant_id
                    }
                )
                
                start_time = time.time()
                
                # Execute tool via MCP client
                tool_result = await self.mcp_client.execute_tool(
                    tool_name=tool_call.name,
                    parameters=tool_call.parameters
                )
                
                execution_time = time.time() - start_time
                
                # Create tool result
                result = ToolCallResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=tool_result,
                    execution_time=execution_time,
                    status="success"
                )
                
                tool_results.append(result)
                
                # Update tool span
                tool_span.end(
                    output=tool_result,
                    metadata={
                        "execution_time": execution_time,
                        "status": "success"
                    }
                )
                
                logger.info(f"Tool {tool_call.name} executed successfully for tenant {self.tenant_id}")
                
            except Exception as e:
                logger.error(f"Tool {tool_call.name} execution failed for tenant {self.tenant_id}: {e}")
                
                error_result = ToolCallResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result={},
                    execution_time=0,
                    status="error",
                    error=str(e)
                )
                
                tool_results.append(error_result)
                
                # Update tool span with error
                if 'tool_span' in locals():
                    tool_span.end(
                        metadata={
                            "status": "error",
                            "error": str(e)
                        }
                    )
        
        # If tools were executed, generate a follow-up response incorporating the results
        if tool_results:
            response_text = await self._generate_follow_up_response(
                original_response=response_text,
                tool_results=tool_results
            )
        
        return response_text, tool_results
    
    def _extract_tool_calls_from_response(self, response_text: str) -> List[ToolCall]:
        """Extract tool calls from LLM response (simplified implementation)"""
        # This is a simplified implementation. In practice, you'd need more
        # sophisticated parsing based on your LLM's response format
        tool_calls = []
        
        # Look for function call patterns in the response
        # This would depend on how your LLM formats tool calls
        # For example, looking for JSON-formatted function calls
        
        return tool_calls
    
    async def _generate_follow_up_response(
        self,
        original_response: str,
        tool_results: List[ToolCallResult]
    ) -> str:
        """Generate a follow-up response incorporating tool results"""
        # Build context from tool results
        tool_context = []
        for result in tool_results:
            if result.status == "success":
                tool_context.append(f"Tool {result.name} returned: {result.result}")
            else:
                tool_context.append(f"Tool {result.name} failed: {result.error}")
        
        # Create follow-up prompt
        follow_up_messages = [
            {"role": "system", "content": self.tenant_config.system_prompt},
            {"role": "user", "content": "Please provide a response incorporating the following tool results:"},
            {"role": "assistant", "content": "\n".join(tool_context)}
        ]
        
        # Generate enhanced response
        enhanced_response = await self._generate_llm_response(
            messages=follow_up_messages,
            brief_mode=False,
            available_tools=[]
        )
        
        return enhanced_response
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        if not self.mcp_client:
            return []
        
        tools = self.mcp_client.get_available_tools()
        return [tool["function"]["name"] for tool in tools]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the agent and its components"""
        health_data = {
            "tenant_id": self.tenant_id,
            "status": "healthy",
            "tenant_config_loaded": self.tenant_config is not None,
            "llm_service_initialized": self.llm_service is not None,
            "tools_enabled": False,
            "mcp_connection": None,
            "available_tools": 0
        }
        
        if self.tenant_config:
            health_data["tools_enabled"] = self.tenant_config.tools.enabled
            
            if self.mcp_client:
                mcp_health = await self.mcp_client.health_check()
                health_data["mcp_connection"] = mcp_health
                health_data["available_tools"] = len(self.get_available_tools())
        
        return health_data 