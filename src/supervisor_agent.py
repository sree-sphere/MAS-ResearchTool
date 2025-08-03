"""
Supervisor Agent - Orchestrates the intelligent query routing workflow
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

from src.cache_manager import CacheManager
from src.rag_agent import RAGAgent
from src.multi_agent_system import ResearchPipeline
from src.models.models import ResearchRequest
from src.log import logger


class QueryRouting(BaseModel):
    similarity_score: float
    should_use_cache: bool
    should_use_rag: bool
    should_run_pipeline: bool
    reasoning: str


class SupervisorAgent:
    """Supervisor agent that orchestrates the query routing workflow"""
    
    def __init__(self, llm_provider: str = "openai"):
        self.llm = self._initialize_llm(llm_provider)
        self.cache_manager = CacheManager()
        self.rag_agent = RAGAgent()
        self.research_pipeline = ResearchPipeline(llm_provider)
        
    def _initialize_llm(self, provider: str):
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            temperature=0.1,
            max_tokens=1000
        )
    
    async def check_query_similarity(self, query: str) -> Dict:
        """Check similarity with cached queries"""
        try:
            similar_queries = await self.cache_manager.find_similar_queries(query)
            if not similar_queries:
                return {"max_similarity": 0.0, "matches": []}
            
            max_similarity = max(q["similarity"] for q in similar_queries)
            return {
                "max_similarity": max_similarity,
                "matches": similar_queries[:3]
            }
        except Exception as e:
            logger.error(f"Error checking query similarity: {e}")
            return {"max_similarity": 0.0, "matches": []}
    
    async def route_query_decision(self, query: str, similarity_data: Dict) -> QueryRouting:
        """Make routing decision based on similarity"""
        max_similarity = similarity_data.get("max_similarity", 0.0)
        
        prompt = ChatPromptTemplate.from_template(
            """Based on the query similarity analysis, determine the best routing strategy.
            
            Query: {query}
            Max Similarity Score: {similarity}
            Similar Queries: {matches}
            
            Routing Rules:
            - ≥90% similarity: Use direct cache
            - 80-89% similarity: Use RAG retrieval  
            - <80% similarity: Run full pipeline
            
            Consider:
            - Query intent and specificity
            - Information freshness requirements
            - Computational efficiency
            
            Provide reasoning and route decision."""
        )
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                query=query,
                similarity=max_similarity,
                matches=similarity_data.get("matches", [])
            )
        )
        
        # Parse decision
        should_use_cache = max_similarity >= 0.90
        should_use_rag = 0.80 <= max_similarity < 0.90
        should_run_pipeline = max_similarity < 0.80
        
        return QueryRouting(
            similarity_score=max_similarity,
            should_use_cache=should_use_cache,
            should_use_rag=should_use_rag,
            should_run_pipeline=should_run_pipeline,
            reasoning=response.content
        )
    
    async def execute_workflow(self, query: str, request: ResearchRequest) -> Dict:
        """Execute the complete intelligent routing workflow"""
        workflow_start = datetime.now()
        
        try:
            logger.info(f"Supervisor starting workflow for query: {query[:100]}...")
            
            # Step 1: Check query similarity
            similarity_data = await self.check_query_similarity(query)
            routing_decision = await self.route_query_decision(query, similarity_data)
            
            logger.info(f"Routing decision: {routing_decision.reasoning}")
            
            # Step 2: Route based on decision
            if routing_decision.should_use_cache:
                logger.info("Using direct cache retrieval")
                result = await self._handle_cache_route(query, similarity_data)
                
            elif routing_decision.should_use_rag:
                logger.info("Using RAG retrieval")
                result = await self._handle_rag_route(query, request)
                
            else:
                logger.info("Running full research pipeline")
                result = await self._handle_pipeline_route(query, request)
            
            # Step 3: Cache new unique queries
            if result.get("status") == "completed":
                await self._cache_if_unique(query, result)
            
            execution_time = (datetime.now() - workflow_start).total_seconds()
            result["execution_time"] = execution_time
            result["routing_decision"] = routing_decision.dict()
            
            logger.info(f"Workflow completed in {execution_time:.2f}s via {result.get('source', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Supervisor workflow failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "source": "supervisor_error"
            }
    
    async def _handle_cache_route(self, query: str, similarity_data: Dict) -> Dict:
        """Handle direct cache retrieval"""
        try:
            best_match = similarity_data["matches"][0]
            cached_result = await self.cache_manager.get_cached_result(best_match["query"])
            
            if cached_result:
                # Store new query mapping to same result
                await self.cache_manager.store_query_mapping(query, cached_result["chroma_id"])
                
                return {
                    "status": "completed",
                    "source": "cache",
                    "similarity_used": best_match["similarity"],
                    "original_query": best_match["query"],
                    "generated_content": cached_result["content"],
                    "metadata": cached_result.get("metadata", {})
                }
            else:
                # Cache miss, fallback to RAG
                logger.warning("Cache miss, falling back to RAG")
                request = ResearchRequest(topic=query, content_types=["summary"])
                return await self._handle_rag_route(query, request)
                
        except Exception as e:
            logger.error(f"Cache route failed: {e}")
            raise
    
    async def _handle_rag_route(self, query: str, request: ResearchRequest) -> Dict:
        """Handle RAG retrieval route"""
        try:
            # Get top-3 relevant documents
            rag_results = await self.rag_agent.retrieve_and_generate(query, top_k=3)
            
            # Check if RAG results answer the query
            if await self._validate_rag_answer(query, rag_results):
                # Store unique query
                await self._cache_if_unique(query, {
                    "generated_content": rag_results["generated_content"],
                    "source": "rag"
                })
                
                return {
                    "status": "completed",
                    "source": "rag",
                    "generated_content": rag_results["generated_content"],
                    "retrieved_docs": rag_results["retrieved_docs"],
                    "confidence": rag_results["confidence"]
                }
            else:
                # RAG insufficient, run full pipeline
                logger.info("RAG results insufficient, running full pipeline")
                return await self._handle_pipeline_route(query, request)
                
        except Exception as e:
            logger.error(f"RAG route failed: {e}")
            # Fallback to pipeline
            return await self._handle_pipeline_route(query, request)
    
    async def _handle_pipeline_route(self, query: str, request: ResearchRequest) -> Dict:
        """Handle full research pipeline route"""
        try:
            pipeline_id = f"supervisor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Run full pipeline
            result = await self.research_pipeline.execute_pipeline(pipeline_id, request)
            
            if result.get("status") == "completed":
                # Store in ChromaDB and cache
                chroma_id = await self.rag_agent.store_pipeline_results(query, result)
                await self.cache_manager.store_query_result(query, chroma_id, result)
                
                result["source"] = "pipeline"
                result["chroma_id"] = chroma_id
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline route failed: {e}")
            raise
    
    async def _validate_rag_answer(self, query: str, rag_results: Dict) -> bool:
        """Validate if RAG results adequately answer the query"""
        prompt = ChatPromptTemplate.from_template(
            """Evaluate if this generated content adequately answers the user's query.
            
            Query: {query}
            Generated Content: {content}
            
            Consider:
            - Completeness of answer
            - Relevance to query
            - Factual coherence
            - Sufficient detail level
            
            Respond with only: YES or NO"""
        )
        
        try:
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    query=query,
                    content=rag_results.get("generated_content", [{}])[0].get("content", "")[:1000]
                )
            )
            return response.content.strip().upper() == "YES"
        except:
            return False
    
    async def _cache_if_unique(self, query: str, result: Dict):
        """Cache query if it's unique enough"""
        try:
            is_unique = await self.cache_manager.is_query_unique(query, threshold=0.95)
            if is_unique:
                if result.get("source") == "rag":
                    # For RAG results, create new ChromaDB entry
                    chroma_id = await self.rag_agent.store_rag_results(query, result)
                    await self.cache_manager.store_query_result(query, chroma_id, result)
                # Pipeline results already stored in _handle_pipeline_route
                logger.info(f"Cached unique query: {query[:50]}...")
        except Exception as e:
            logger.warning(f"Failed to cache query: {e}")