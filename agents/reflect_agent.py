import logging
from typing import Dict, Any, Optional, List, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langfuse import Langfuse
from tavily import TavilyClient

from config.settings import settings
from services.tavily_search import TavilySearchService
from services.rag_service import RAGService

logger = logging.getLogger(__name__)

class AgentState(Dict):
    """State of the reflect agent"""
    messages: List[dict]
    current_message: str
    thought: str
    action: Optional[str]
    action_input: Optional[str]
    observation: str
    final_answer: str
    needs_action: bool
    iteration_count: int
    max_iterations: int

class ReflectAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.google_ai_generative,
            temperature=0.7
        )
        
        # Initialize services
        self.tavily_service = TavilySearchService()
        self.rag_service = RAGService()
        
        # Initialize Langfuse for observability
        self.langfuse = Langfuse(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.langfuse_public_key,
            host=settings.langfuse_host
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph with conditional edges"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("think", self._think_node)
        workflow.add_node("act", self._act_node)
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("respond", self._final_answer_node)
        
        # Set entry point
        workflow.set_entry_point("think")
        
        # Add conditional edges - this addresses the manager's feedback
        workflow.add_conditional_edges(
            "think",
            self._should_continue,
            {
                "act": "act",
                "final_answer": "respond"
            }
        )
        
        workflow.add_edge("act", "observe")
        
        # After observation, decide whether to continue thinking or finish
        workflow.add_conditional_edges(
            "observe",
            self._should_continue_after_observation,
            {
                "think": "think",
                "final_answer": "respond"
            }
        )
        
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def _think_node(self, state: AgentState) -> AgentState:
        """Thinking node - analyze the situation and decide what to do"""
        logger.info("Entering think node")
        
        # Increment iteration count
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        system_prompt = """You are a helpful AI assistant that thinks step by step.
        
        Given the user's message and any previous observations, think about how to respond.
        
        Decide if you need to:
        1. Search for current information using web search
        2. Search through documents using RAG
        3. Provide a direct answer based on your knowledge
        
        Format your response as:
        THOUGHT: [your reasoning about what to do]
        ACTION: [search_web|search_documents|direct_answer]
        ACTION_INPUT: [query for search or direct answer]
        """
        
        # Build context from previous observations
        context = ""
        if state.get("observation"):
            context = f"Previous observation: {state['observation']}\n"
        
        prompt = f"{system_prompt}\n\nUser message: {state['current_message']}\n{context}"
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Parse the response
            thought = ""
            action = None
            action_input = None
            
            lines = content.split('\n')
            for line in lines:
                if line.startswith("THOUGHT:"):
                    thought = line.replace("THOUGHT:", "").strip()
                elif line.startswith("ACTION:"):
                    action = line.replace("ACTION:", "").strip()
                elif line.startswith("ACTION_INPUT:"):
                    action_input = line.replace("ACTION_INPUT:", "").strip()
            
            state["thought"] = thought
            state["action"] = action
            state["action_input"] = action_input
            
            # Determine if action is needed
            state["needs_action"] = action in ["search_web", "search_documents"]
            
            logger.info(f"Thought: {thought}")
            logger.info(f"Action: {action}")
            logger.info(f"Needs action: {state['needs_action']}")
            
        except Exception as e:
            logger.error(f"Error in think node: {e}")
            state["thought"] = "I encountered an error while thinking."
            state["needs_action"] = False
        
        return state
    
    def _act_node(self, state: AgentState) -> AgentState:
        """Action node - perform the decided action"""
        logger.info("Entering act node")
        
        action = state.get("action")
        action_input = state.get("action_input", "")
        
        try:
            if action == "search_web":
                result = self.tavily_service.search(action_input)
                state["observation"] = f"Web search results: {result}"
                
            elif action == "search_documents":
                result = self.rag_service.search(action_input)
                state["observation"] = f"Document search results: {result}"
                
            else:
                state["observation"] = "No action taken."
                
        except Exception as e:
            logger.error(f"Error in act node: {e}")
            state["observation"] = f"Error performing action: {str(e)}"
        
        return state
    
    def _observe_node(self, state: AgentState) -> AgentState:
        """Observation node - process the results of the action"""
        logger.info("Entering observe node")
        
        # The observation is already set in the act node
        # This node can be used for additional processing if needed
        logger.info(f"Observation: {state.get('observation', 'No observation')}")
        
        return state
    
    def _final_answer_node(self, state: AgentState) -> AgentState:
        """Generate the final answer"""
        logger.info("Entering final answer node")
        
        system_prompt = """You are a helpful AI assistant. Based on the user's message, your thoughts, and any observations you've made, provide a helpful and accurate response.
        
        Be conversational and helpful. If you searched for information, incorporate it naturally into your response.
        """
        
        # Build context
        context_parts = []
        if state.get("thought"):
            context_parts.append(f"My thinking: {state['thought']}")
        if state.get("observation"):
            context_parts.append(f"Information found: {state['observation']}")
        
        context = "\n".join(context_parts)
        prompt = f"{system_prompt}\n\nUser message: {state['current_message']}\n\n{context}\n\nResponse:"
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["final_answer"] = response.content
            
        except Exception as e:
            logger.error(f"Error generating final answer: {e}")
            state["final_answer"] = "I apologize, but I encountered an error while generating my response."
        
        return state
    
    def _should_continue(self, state: AgentState) -> Literal["act", "final_answer"]:
        """Conditional edge: decide whether to act or provide final answer"""
        # Check iteration limit
        max_iterations = state.get("max_iterations", 3)
        current_iteration = state.get("iteration_count", 0)
        
        if current_iteration >= max_iterations:
            logger.info("Reached maximum iterations, providing final answer")
            return "final_answer"
        
        # Check if action is needed
        if state.get("needs_action", False):
            logger.info("Action needed, proceeding to act")
            return "act"
        else:
            logger.info("No action needed, providing final answer")
            return "final_answer"
    
    def _should_continue_after_observation(self, state: AgentState) -> Literal["think", "final_answer"]:
        """Conditional edge: decide whether to think more or provide final answer"""
        max_iterations = state.get("max_iterations", 3)
        current_iteration = state.get("iteration_count", 0)
        
        if current_iteration >= max_iterations:
            logger.info("Reached maximum iterations after observation, providing final answer")
            return "final_answer"
        
        # For now, we'll provide the final answer after one observation
        # This can be made more sophisticated based on the observation
        return "final_answer"
    
    async def run(self, message: str, session_id: Optional[str] = None) -> str:
        """Run the reflect agent"""
        logger.info(f"Running reflect agent for message: {message}")
        
        # Initialize state
        initial_state = AgentState(
            messages=[],
            current_message=message,
            thought="",
            action=None,
            action_input=None,
            observation="",
            final_answer="",
            needs_action=False,
            iteration_count=0,
            max_iterations=3
        )
        
        try:
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            logger.info("Reflect agent completed successfully")
            return result["final_answer"]
            
        except Exception as e:
            logger.error(f"Error running reflect agent: {e}")
            return "I apologize, but I encountered an error while processing your request." 