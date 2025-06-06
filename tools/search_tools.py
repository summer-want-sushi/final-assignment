# tools/search_tools.py
import requests
import os
from typing import List, Dict, Any, Optional
from .utils import get_env_var, logger

class SearchTools:
    """Free and cost-effective search tools with multiple providers"""
    
    def __init__(self):
        # Primary: Free alternatives
        self.duckduckgo_enabled = True
        
        # Secondary: Tavily (cost-effective)
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Tertiary: SerpAPI (expensive, fallback only)
        self.serpapi_key = os.getenv("SERPAPI_KEY")
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Free search using DuckDuckGo Instant Answer API
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            # DuckDuckGo Instant Answer API (free)
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process abstract
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', 'DuckDuckGo Result'),
                    'url': data.get('AbstractURL', ''),
                    'content': data.get('Abstract', ''),
                    'source': 'DuckDuckGo'
                })
            
            # Process related topics
            for topic in data.get('RelatedTopics', [])[:max_results-len(results)]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'url': topic.get('FirstURL', ''),
                        'content': topic.get('Text', ''),
                        'source': 'DuckDuckGo'
                    })
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {str(e)}")
            return []
    
    def search_tavily(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using Tavily API (cost-effective)
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        if not self.tavily_api_key:
            logger.warning("Tavily API key not provided")
            return []
        
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": False,
                "include_images": False,
                "include_raw_content": False,
                "max_results": max_results
            }
            
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for result in data.get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'source': 'Tavily'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Tavily search failed: {str(e)}")
            return []
    
    def search_serpapi(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search using SerpAPI (expensive, fallback only)
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results
        """
        if not self.serpapi_key:
            logger.warning("SerpAPI key not provided")
            return []
        
        try:
            url = "https://serpapi.com/search"
            params = {
                'api_key': self.serpapi_key,
                'engine': 'google',
                'q': query,
                'num': max_results,
                'gl': 'us',  # Geolocation
                'hl': 'en'   # Language
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            for result in data.get('organic_results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'content': result.get('snippet', ''),
                    'source': 'Google (SerpAPI)'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {str(e)}")
            return []
    
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Comprehensive search using multiple providers with fallback strategy
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of search results from best available provider
        """
        if not query.strip():
            return []
        
        # Try providers in order of preference (free -> cheap -> expensive)
        providers = [
            ("DuckDuckGo", self.search_duckduckgo),
            ("Tavily", self.search_tavily),
            ("SerpAPI", self.search_serpapi)
        ]
        
        for provider_name, search_func in providers:
            try:
                logger.info(f"Attempting search with {provider_name}")
                results = search_func(query, max_results)
                
                if results:
                    logger.info(f"Successfully retrieved {len(results)} results from {provider_name}")
                    return results
                else:
                    logger.warning(f"No results from {provider_name}")
                    
            except Exception as e:
                logger.error(f"Error with {provider_name}: {str(e)}")
                continue
        
        logger.error("All search providers failed")
        return []
    
    def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for news articles"""
        news_query = f"news {query}"
        return self.search(news_query, max_results)
    
    def search_academic(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for academic content"""
        academic_query = f"academic research {query} site:scholar.google.com OR site:arxiv.org OR site:researchgate.net"
        return self.search(academic_query, max_results)

# Convenience functions
def search_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Standalone function for web search"""
    tools = SearchTools()
    return tools.search(query, max_results)

def search_news(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Standalone function for news search"""
    tools = SearchTools()
    return tools.search_news(query, max_results)
