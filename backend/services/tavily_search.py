import logging
from typing import List, Dict, Any, Optional
from tavily import TavilyClient

from config.settings import settings

logger = logging.getLogger(__name__)

class TavilySearchService:
    def __init__(self):
        self.client = TavilyClient(api_key=settings.tavily_api_key)
    
    def search(self, query: str, max_results: int = 5) -> str:
        """Search the web using Tavily"""
        try:
            logger.info(f"Searching with Tavily: {query}")
            
            # Perform the search
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            
            # Extract relevant information
            results = []
            
            # Add the direct answer if available
            if response.get("answer"):
                results.append(f"Direct Answer: {response['answer']}")
            
            # Add search results
            if response.get("results"):
                for i, result in enumerate(response["results"][:max_results], 1):
                    title = result.get("title", "")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    
                    result_text = f"{i}. {title}\n{content}"
                    if url:
                        result_text += f"\nSource: {url}"
                    
                    results.append(result_text)
            
            if not results:
                return "No relevant information found."
            
            formatted_results = "\n\n".join(results)
            logger.info(f"Tavily search completed, found {len(results)} results")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching with Tavily: {e}")
            return f"Error performing web search: {str(e)}"
    
    def get_answer(self, query: str) -> str:
        """Get a direct answer using Tavily"""
        try:
            logger.info(f"Getting answer from Tavily: {query}")
            
            response = self.client.qna_search(query=query)
            
            if response:
                return response
            else:
                return "No direct answer available."
                
        except Exception as e:
            logger.error(f"Error getting answer from Tavily: {e}")
            return f"Error getting answer: {str(e)}" 