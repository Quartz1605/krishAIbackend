"""
RAG Chatbot Service for integrating with existing agricultural chatbot
"""
import asyncio
import aiohttp
from typing import Optional, Dict, Any
import json
from datetime import datetime

from ..utils.logger import logger
from ..utils.helpers import async_retry
from ..config.settings import settings


class RAGChatbotService:
    """
    Service for integrating with existing RAG-based agricultural chatbot
    """
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize RAG chatbot service
        
        Args:
            api_url: Optional API URL override
            api_key: Optional API key override
        """
        self.api_url = api_url or settings.RAG_API_URL
        self.api_key = api_key or settings.RAG_API_KEY
        self.timeout = settings.RAG_TIMEOUT
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Context management for conversations
        self.conversation_context: Dict[str, list] = {}
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    @async_retry(max_attempts=3, delay=1.0)
    async def query_chatbot(
        self,
        query: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG chatbot with agricultural question
        
        Args:
            query: English query text
            user_id: Optional user identifier for context
            context: Optional additional context
            
        Returns:
            Dictionary with response and metadata
        """
        await self._ensure_session()
        
        try:
            # Prepare request payload
            payload = {
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
                "language": "en"
            }
            
            if user_id:
                payload["user_id"] = user_id
            
            if context:
                payload["context"] = context
            
            # Add conversation history if available
            if user_id and user_id in self.conversation_context:
                payload["conversation_history"] = self.conversation_context[user_id][-5:]  # Last 5 interactions
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make request to RAG chatbot
            async with self.session.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Store in conversation context
                    if user_id:
                        if user_id not in self.conversation_context:
                            self.conversation_context[user_id] = []
                        
                        self.conversation_context[user_id].append({
                            "query": query,
                            "response": result.get("response", ""),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    
                    logger.info(f"RAG chatbot query successful for: {query[:50]}...")
                    
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "confidence": result.get("confidence", 0.0),
                        "sources": result.get("sources", []),
                        "metadata": result.get("metadata", {})
                    }
                
                else:
                    error_text = await response.text()
                    logger.error(f"RAG chatbot error (status {response.status}): {error_text}")
                    
                    return {
                        "success": False,
                        "response": "I'm sorry, I couldn't process your agricultural query at this time.",
                        "error": error_text
                    }
                    
        except asyncio.TimeoutError:
            logger.error(f"RAG chatbot timeout for query: {query[:50]}...")
            return {
                "success": False,
                "response": "The request took too long to process. Please try again.",
                "error": "Timeout"
            }
            
        except Exception as e:
            logger.error(f"RAG chatbot error: {str(e)}")
            return {
                "success": False,
                "response": "An error occurred while processing your query.",
                "error": str(e)
            }
    
    async def query_with_fallback(self, query: str, user_id: Optional[str] = None) -> str:
        """
        Query chatbot with fallback response
        
        Args:
            query: English query text
            user_id: Optional user identifier
            
        Returns:
            Response text (never None)
        """
        result = await self.query_chatbot(query, user_id)
        
        if result["success"] and result["response"]:
            return result["response"]
        
        # Fallback responses for common agricultural queries
        fallback_responses = self._get_fallback_response(query)
        return fallback_responses
    
    def _get_fallback_response(self, query: str) -> str:
        """
        Get fallback response for common agricultural queries
        
        Args:
            query: User query
            
        Returns:
            Fallback response
        """
        query_lower = query.lower()
        
        # Common agricultural query patterns and responses
        if any(word in query_lower for word in ["disease", "pest", "infection"]):
            return ("I understand you're asking about crop diseases or pests. "
                   "Please consult with local agricultural experts for accurate diagnosis. "
                   "Common signs include yellowing leaves, spots, or unusual growth patterns.")
        
        elif any(word in query_lower for word in ["fertilizer", "nutrient", "manure"]):
            return ("For fertilizer recommendations, consider soil testing first. "
                   "Generally, balanced NPK fertilizers work well for most crops. "
                   "Organic options include compost and well-rotted manure.")
        
        elif any(word in query_lower for word in ["weather", "rain", "monsoon"]):
            return ("Weather conditions are crucial for farming. "
                   "Check local weather forecasts regularly and plan your farming activities accordingly. "
                   "Consider using weather-resistant crop varieties.")
        
        elif any(word in query_lower for word in ["irrigation", "water", "watering"]):
            return ("Proper irrigation is essential for crop health. "
                   "Drip irrigation and sprinkler systems are water-efficient methods. "
                   "Water requirements vary by crop and growth stage.")
        
        elif any(word in query_lower for word in ["seed", "planting", "sowing"]):
            return ("Use quality seeds from reliable sources. "
                   "Follow recommended planting depths and spacing for your crop. "
                   "Consider soil temperature and moisture before sowing.")
        
        elif any(word in query_lower for word in ["harvest", "yield", "production"]):
            return ("Harvest at the right maturity stage for best quality and yield. "
                   "Use proper harvesting techniques to minimize losses. "
                   "Store produce in appropriate conditions.")
        
        else:
            return ("I'm here to help with agricultural queries. "
                   "Please ask about crops, diseases, fertilizers, irrigation, or farming practices. "
                   "For specific advice, consult local agricultural extension services.")
    
    def clear_context(self, user_id: str):
        """
        Clear conversation context for a user
        
        Args:
            user_id: User identifier
        """
        if user_id in self.conversation_context:
            del self.conversation_context[user_id]
            logger.info(f"Cleared context for user: {user_id}")
    
    async def health_check(self) -> bool:
        """
        Check if RAG chatbot service is healthy
        
        Returns:
            True if service is responding, False otherwise
        """
        try:
            await self._ensure_session()
            
            # Try to make a health check request
            health_url = self.api_url.replace("/chat", "/health")
            
            async with self.session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def format_agricultural_response(self, response: str) -> str:
        """
        Format response with agricultural context
        
        Args:
            response: Raw response text
            
        Returns:
            Formatted response
        """
        # Add helpful prefixes/suffixes for agricultural context
        formatted = response
        
        # Add disclaimer if discussing treatments
        if any(word in response.lower() for word in ["pesticide", "chemical", "spray", "medicine"]):
            formatted += "\n\nNote: Always follow label instructions and local regulations when using agricultural chemicals."
        
        # Add consultation reminder for serious issues
        if any(word in response.lower() for word in ["severe", "serious", "emergency", "urgent"]):
            formatted += "\n\nFor serious agricultural issues, please consult local agricultural officers or experts immediately."
        
        return formatted
