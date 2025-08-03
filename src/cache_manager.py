"""
Cache Manager - Handles Redis caching and query similarity
"""

import json
import redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.log import logger


class CacheManager:
    """Manages Redis cache and query similarity matching"""
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_ttl = 48 * 3600  # 48 hours
        
    async def find_similar_queries(self, query: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar queries in cache"""
        try:
            # Get all cached queries
            pattern = "query:*"
            cached_keys = self.redis_client.keys(pattern)
            
            if not cached_keys:
                return []
            
            # Get query embeddings
            query_embedding = self.encoder.encode([query])
            similarities = []
            
            for key in cached_keys:
                try:
                    cached_data = json.loads(self.redis_client.get(key))
                    cached_query = cached_data.get("original_query", "")
                    
                    if cached_query:
                        cached_embedding = self.encoder.encode([cached_query])
                        similarity = cosine_similarity(query_embedding, cached_embedding)[0][0]
                        
                        if similarity >= threshold:
                            similarities.append({
                                "query": cached_query,
                                "similarity": float(similarity),
                                "cached_at": cached_data.get("cached_at"),
                                "chroma_id": cached_data.get("chroma_id")
                            })
                except Exception as e:
                    logger.warning(f"Error processing cached key {key}: {e}")
                    continue
            
            # Sort by similarity descending
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:10]  # Return top 10
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []
    
    async def get_cached_result(self, query: str) -> Optional[Dict]:
        """Get cached result for exact or mapped query"""
        try:
            query_key = f"query:{hash(query)}"
            cached_data = self.redis_client.get(query_key)
            
            if cached_data:
                return json.loads(cached_data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    async def store_query_result(self, query: str, chroma_id: str, result: Dict):
        """Store query result in cache"""
        try:
            query_key = f"query:{hash(query)}"
            
            cache_data = {
                "original_query": query,
                "chroma_id": chroma_id,
                "content": result.get("generated_content", []),
                "metadata": {
                    "cached_at": datetime.now().isoformat(),
                    "source": result.get("source", "unknown"),
                    "execution_time": result.get("execution_time", 0)
                }
            }
            
            self.redis_client.setex(
                query_key,
                self.cache_ttl,
                json.dumps(cache_data, default=str)
            )
            
            logger.info(f"Stored query in cache: {query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing query result: {e}")
    
    async def store_query_mapping(self, new_query: str, chroma_id: str):
        """Store mapping of new query to existing ChromaDB result"""
        try:
            query_key = f"query:{hash(new_query)}"
            
            # Get existing result from another query with same chroma_id
            existing_data = await self._get_data_by_chroma_id(chroma_id)
            
            if existing_data:
                cache_data = {
                    "original_query": new_query,
                    "chroma_id": chroma_id,
                    "content": existing_data["content"],
                    "metadata": {
                        "cached_at": datetime.now().isoformat(),
                        "source": "mapping",
                        "mapped_from": existing_data["metadata"].get("original_query")
                    }
                }
                
                self.redis_client.setex(
                    query_key,
                    self.cache_ttl,
                    json.dumps(cache_data, default=str)
                )
                
                logger.info(f"Stored query mapping: {new_query[:50]}...")
            
        except Exception as e:
            logger.error(f"Error storing query mapping: {e}")
    
    async def is_query_unique(self, query: str, threshold: float = 0.95) -> bool:
        """Check if query is unique enough to warrant caching"""
        try:
            similar_queries = await self.find_similar_queries(query, threshold)
            return len(similar_queries) == 0
        except:
            return True  # Default to unique if check fails
    
    async def _get_data_by_chroma_id(self, chroma_id: str) -> Optional[Dict]:
        """Helper to get cached data by ChromaDB ID"""
        try:
            pattern = "query:*"
            for key in self.redis_client.keys(pattern):
                data = json.loads(self.redis_client.get(key))
                if data.get("chroma_id") == chroma_id:
                    return data
            return None
        except:
            return None
    
    def cleanup_expired(self):
        """Manual cleanup of expired entries (Redis handles TTL automatically)"""
        try:
            pattern = "query:*"
            expired_count = 0
            
            for key in self.redis_client.keys(pattern):
                ttl = self.redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist
                    expired_count += 1
            
            logger.info(f"Cleaned up {expired_count} expired cache entries")
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")