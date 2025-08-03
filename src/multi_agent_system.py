"""
Multi-Agent System using LangGraph and LangChain
Implements a research and content generation pipeline with multiple AI agents.

This system demonstrates:
- Agent orchestration with LangGraph
- Complex state management
- Inter-agent communication
- Error handling and recovery
- Performance monitoring
"""

import asyncio
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from duckduckgo_search import DDGS
from langchain_community.utilities import WikipediaAPIWrapper
from src.models.models import ResearchRequest, AgentMetrics, ResearchResult, ContentOutput
from typing_extensions import TypedDict
from langchain_core.tools import tool


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AgentRole(str, Enum):
    """Agent roles in the system"""
    RESEARCHER = "researcher"
    FACT_CHECKER = "fact_checker"
    CONTENT_GENERATOR = "content_generator"
    QUALITY_ASSURANCE = "quality_assurance"
    COORDINATOR = "coordinator"

class PipelineStatus(str, Enum):
    """Pipeline execution status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PipelineState(TypedDict):
    """State shared between agents in the pipeline"""
    request: ResearchRequest
    research_results: List[ResearchResult]
    fact_check_results: Dict[str, Any]
    generated_content: List[ContentOutput]
    current_step: str
    progress: int
    errors: List[str]
    agent_outputs: Dict[str, Any]
    execution_start: datetime
    metadata: Dict[str, Any]

class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, role: AgentRole, llm_provider: str = "openai"):
        self.role = role
        self.agent_id = f"{role.value}_{id(self)}"
        self.metrics = AgentMetrics(agent_id=self.agent_id, role=role)
        self.llm_provider = llm_provider
        self.llm = self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the LLM based on provider"""
        if self.llm_provider == "openai":
            return ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=0.3,
                max_tokens=2000,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "")
            )
        elif self.llm_provider == "anthropic":
            return ChatAnthropic(
                model=os.getenv("ANTHROPIC_MODEL", "claude-2"),
                temperature=0.3,
                max_tokens=2000
            )
        else:
            # Fallback
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    def _update_metrics(self, success: bool, execution_time: float):
        """Update agent performance metrics"""
        self.metrics.total_executions += 1
        if success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
        
        # Calculate running average
        if self.metrics.total_executions == 1:
            self.metrics.avg_execution_time = execution_time
        else:
            self.metrics.avg_execution_time = (self.metrics.avg_execution_time * (self.metrics.total_executions - 1) + execution_time) / self.metrics.total_executions
        
        self.metrics.last_execution = datetime.now()

class ResearchAgent(BaseAgent):
    """Agent responsible for gathering information from various sources"""
    
    def __init__(self, llm_provider: str = "openai"):
        super().__init__(AgentRole.RESEARCHER, llm_provider)
        # Search tools init
        self.ddgs = DDGS()
        self.wikipedia_tool = WikipediaAPIWrapper(top_k_results=3)
        self.max_results_per_query = 3
        self.search_backends = ["text", "news"]  # Multiple DuckDuckGo backends upon testing
        self.backoff_times = [1, 2, 4]  # Exponential backoff times for search retries
        
    
    async def execute(self, input: dict) -> dict:
        """Execute research tasks"""
        state: PipelineState = input
        start_time = time.time()
        self.metrics.current_status = "researching"
        
        try:
            request = state["request"]
            if not request:
                raise ValueError("No request found in state")
            # Handle both dict and object formats
            if isinstance(request, dict):
                topic = request.get("topic", "Unknown Topic")
                depth = request.get("depth", "standard")
                max_sources = request.get("max_sources", 5)
            else:
                topic = getattr(request, 'topic', "Unknown Topic")
                depth = getattr(request, 'depth', "standard") 
                max_sources = getattr(request, 'max_sources', 5)
            logger.info(f"Research Agent starting research on: {topic}")

            # Generate research queries
            queries = await self._generate_research_queries(topic, depth)
            
            # Perform searches with retry logic
            research_results = []
            for i, query in enumerate(queries[:max_sources]):
                try:
                    # Web search with multiple backends
                    search_results = await self._web_search_with_retry(query)
                    research_results.extend(search_results)
                    
                    # Wikipedia search only for entity-based queries
                    if i < 3 and await self._is_entity_query(query):
                        wiki_results = await self._wikipedia_search(query)
                        research_results.extend(wiki_results)
                    elif i < 3:
                        # Fallback to Wikipedia for user's topic if query is not entity-based
                        logger.info(f"Query '{query}' not entity-based, performing Wikipedia search")
                        wiki_results = await self._wikipedia_search(topic)
                        research_results.extend(wiki_results)
                        
                except Exception as e:
                    logger.warning(f"Search failed for query '{query}': {str(e)}")
                    state["errors"].append(f"Search error: {str(e)}")
            
            # Rank and filter results
            ranked_results = await self._rank_results(research_results, request.topic if hasattr(request, 'topic') else request.get('topic', topic))
            
            state["research_results"] = ranked_results[:max_sources]
            state["current_step"] = "research_completed"
            state["progress"] = 25
            
            self._update_metrics(True, time.time() - start_time)
            logger.info(f"Research completed with {len(ranked_results)} results")
            self.metrics.current_status = "idle"
            
            logger.info(f"Research completed: {len(ranked_results)} results found")
            
        except Exception as e:
            logger.error(f"Research Agent failed: {str(e)}")
            state["errors"].append(f"Research failed: {str(e)}")
            self._update_metrics(False, time.time() - start_time)
            self.metrics.current_status = "error"
        
        return state
    
    async def _web_search_with_retry(self, query: str) -> List[ResearchResult]:
        """Perform web search with retry logic and multiple backends"""
        max_retries = 3
        parsed_results = []
        
        for attempt in range(max_retries):
            try:
                # Remove quotes for better search results
                clean_query = query.replace('"', '')
                
                # Rotate through different search backends
                backend = self.search_backends[attempt % len(self.search_backends)]
                logger.info(f"Searching ({backend}) for: {clean_query} (attempt {attempt+1}/{max_retries})")
                
                if backend == "text":
                    search_results = list(self.ddgs.text(
                        keywords=clean_query,
                        max_results=self.max_results_per_query,
                        safesearch='moderate'
                    ))
                elif backend == "news":
                    search_results = list(self.ddgs.news(keywords=clean_query, max_results=self.max_results_per_query))
                
                # Parse results
                parsed_results = []
                for result in search_results:
                    # Handle different result structures
                    title = result.get('title') or result.get('heading') or "Untitled"
                    content = result.get('body') or result.get('text') or result.get('snippet') or result.get('excerpt') or ""
                    url = result.get('href') or result.get('url') or ""
                    
                    if title and content and url:
                        parsed_results.append(ResearchResult(
                            title=title[:200],
                            content=content[:1000],
                            source=url,
                            credibility_score=0.7,
                            relevance_score=0.8
                        ))
                
                if parsed_results:
                    logger.info(f"Found {len(parsed_results)} results via {backend}")
                    return parsed_results
                
            except Exception as e:
                logger.warning(f"Search attempt {attempt+1} failed: {str(e)}")
                await asyncio.sleep(self.backoff_times[attempt])  # Exponential backoff
        
        logger.warning(f"All search attempts failed for query: {query}")
        return []
    
    async def _is_entity_query(self, query: str) -> bool:
        """Determine if a query is suitable for Wikipedia"""
        # Simple heuristic before using LLM
        if any(term in query.lower() for term in ["what", "how", "why", "when", "where"]):
            return False
            
        prompt = ChatPromptTemplate.from_template(
            """Is this query about a specific entity or concept suitable for Wikipedia?
            Query: "{query}"
            
            Respond with only: YES or NO"""
        )
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages(query=query))
            return response.content.strip().upper() == "YES"
        except:
            return False
    
    async def _generate_research_queries(self, topic: str, depth: str) -> List[str]:
        """Generate optimized search queries"""
        prompt = ChatPromptTemplate.from_template(
            """Generate {num_queries} search-optimized queries for: "{topic}"
            
            Guidelines:
            1. Use natural language without quotes
            2. Include specific technical terms
            3. Mix broad overview and specific applications
            4. Avoid question formats
            5. Prioritize recent developments
            
            Examples:
            - Quantum computing applications in cryptography
            - Recent breakthroughs in qubit stability
            - Quantum algorithms for machine learning
            
            Return only the queries, one per line."""
        )
        
        num_queries = {"basic": 3, "standard": 5, "deep": 7, "comprehensive": 10}.get(depth, 5)

        response = await self.llm.ainvoke(prompt.format_messages(topic=topic, depth=depth, num_queries=num_queries))
        queries = [q.strip().replace('"', '') for q in response.content.split('\n') if q.strip()]
        return queries[:num_queries]
    
    async def _web_search(self, query: str) -> List[ResearchResult]:
        """Perform web search using DuckDuckGo"""
        try:
            logger.info(f"Searching web for: {query}")
            
            # Use DDGS directly for more control
            search_results = list(self.ddgs.text(
                keywords=query,
                max_results=self.max_results_per_query,
                safesearch='moderate'
            ))
            
            parsed_results = []
            for result in search_results:
                if result.get('title') and result.get('body') and result.get('href'):
                    parsed_results.append(ResearchResult(
                        title=result['title'],
                        content=result['body'],
                        source=result['href'],
                        credibility_score=0.7,  # Default web credibility
                        relevance_score=0.8
                    ))
            
            logger.info(f"Found {len(parsed_results)} web results for query: {query}")
            return parsed_results
            
        except Exception as e:
            logger.warning(f"Web search failed for query '{query}': {str(e)}")
            return []
    
    async def _wikipedia_search(self, query: str) -> List[ResearchResult]:
        """Search Wikipedia with better parsing"""
        try:
            # Clean query for Wikipedia
            clean_query = query.split(":")[-1].strip()
            logger.info(f"Searching Wikipedia for: {clean_query}")
            
            # Wikipedia wrapper with improved parameters
            self.wikipedia_tool = WikipediaAPIWrapper(
                top_k_results=1,  # Only need 1 good result
                doc_content_chars_max=1500
            )
            summary = await asyncio.to_thread(self.wikipedia_tool.run, clean_query)
            
            if summary:
                return [ResearchResult(
                    title=f"Wikipedia: {clean_query}",
                    content=summary,
                    source="https://en.wikipedia.org",
                    credibility_score=0.9,
                    relevance_score=0.85
                )]
            return []
            
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {str(e)}")
            return []
    
    async def _rank_results(self, results: List[ResearchResult], topic: str) -> List[ResearchResult]:
        """Rank research results by relevance and credibility"""
        if not results:
            return []
        
        # Use LLM to score relevance
        prompt = ChatPromptTemplate.from_template(
            """Rate the relevance of this content to the topic "{topic}" on a scale of 0.0 to 1.0.
            
            Content: {content}
            
            Consider:
            - Direct relevance to the topic
            - Quality and depth of information
            - Factual accuracy apparent in the content
            
            Return only a number between 0.0 and 1.0"""
        )
        
        for result in results:
            try:
                response = await self.llm.ainvoke(prompt.format_messages(topic=topic, content=result.content[:500]))
                relevance = float(response.content.strip())
                result.relevance_score = max(0.0, min(1.0, relevance))
            except Exception as e:
                logger.warning(f"Failed to score relevance for result: {str(e)}")
                # No change to default score
        
        # Sort by combined score
        return sorted(
            results, 
            key=lambda x: (x.relevance_score * 0.7 + x.credibility_score * 0.3), 
            reverse=True
        )

class FactCheckingAgent(BaseAgent):
    """Agent responsible for verifying information accuracy"""
    
    def __init__(self, llm_provider: str = "openai"):
        super().__init__(AgentRole.FACT_CHECKER, llm_provider)
    
    
    async def execute(self, input: dict) -> dict:
        """Execute fact-checking tasks"""
        state: PipelineState = input
        start_time = time.time()
        self.metrics.current_status = "fact_checking"
        
        try:
            research_results = state["research_results"]
            if not research_results:
                raise ValueError("No research results found in state")
            # Handle both dict and object formats
            if isinstance(research_results, dict):
                research_results = [ResearchResult(**r) for r in research_results.get("results", [])]
            elif isinstance(research_results, list) and all(isinstance(r, dict) for r in research_results):
                research_results = [ResearchResult(**r) for r in research_results]
            logger.info(f"Fact-checking {len(research_results)} research results")
            
            fact_check_results = {
                "verified_facts": [],
                "questionable_claims": [],
                "contradictions": [],
                "confidence_scores": {},
                "source_reliability": {}
            }
            
            if not research_results:
                logger.warning("No research results to fact-check")
                state["fact_check_results"] = fact_check_results
                state["current_step"] = "fact_checking_completed"
                state["progress"] = 50
                self._update_metrics(True, time.time() - start_time)
                logger.info("Fact-checking completed with no results")
                self.metrics.current_status = "idle"
                return state
            
            # Extract key claims from research
            claims = await self._extract_claims(research_results)
            
            # Verify claims across sources
            for claim in claims:
                verification = await self._verify_claim(claim, research_results)
                
                if verification["confidence"] > 0.8:
                    fact_check_results["verified_facts"].append(claim)
                elif verification["confidence"] < 0.4:
                    fact_check_results["questionable_claims"].append(claim)
                
                fact_check_results["confidence_scores"][claim] = verification["confidence"]
            
            # Check for contradictions
            contradictions = await self._detect_contradictions(research_results)
            fact_check_results["contradictions"] = contradictions
            
            # Assess source reliability
            for result in research_results:
                reliability = await self._assess_source_reliability(result)
                fact_check_results["source_reliability"][result.source] = reliability
            
            state["fact_check_results"] = fact_check_results
            state["current_step"] = "fact_checking_completed"
            state["progress"] = 50
            
            self._update_metrics(True, time.time() - start_time)
            logger.info(f"Fact-checking completed with {len(fact_check_results['verified_facts'])} verified facts")
            self.metrics.current_status = "idle"
            
            logger.info("Fact-checking completed")
            
        except Exception as e:
            logger.error(f"Fact-checking Agent failed: {str(e)}")
            state["errors"].append(f"Fact-checking failed: {str(e)}")
            self._update_metrics(False, time.time() - start_time)
            self.metrics.current_status = "error"
        
        return state
    
    async def _extract_claims(self, research_results: List[ResearchResult]) -> List[str]:
        """Extract factual claims from research results"""
        if not research_results:
            return []
        
        all_content = "\n\n".join([r.content for r in research_results[:5]])
        
        if not all_content.strip():
            return []
        
        prompt = ChatPromptTemplate.from_template(
            """Extract 5-10 key factual claims from this research content.
            Focus on specific, verifiable statements.
            
            Content: {content}
            
            Return each claim on a separate line, no numbering."""
        )

        response = await self.llm.ainvoke(prompt.format_messages(content=all_content[:3000]))
        
        claims = [c.strip() for c in response.content.split('\n') if c.strip()]
        return claims[:10]
    
    async def _verify_claim(self, claim: str, sources: List[ResearchResult]) -> Dict[str, Any]:
        """Verify a claim against available sources"""
        supporting_sources = 0
        contradicting_sources = 0
        
        for source in sources:
            prompt = ChatPromptTemplate.from_template(
                """Does this source support, contradict, or remain neutral about the claim?
                
                Claim: {claim}
                Source: {content}
                
                Respond with only: SUPPORTS, CONTRADICTS, or NEUTRAL"""
            )
            
            try:
                response = await self.llm.ainvoke(prompt.format_messages(claim=claim, content=source.content[:800]))
                
                verdict = response.content.strip().upper()
                if verdict == "SUPPORTS":
                    supporting_sources += 1
                elif verdict == "CONTRADICTS":
                    contradicting_sources += 1
                    
            except:
                continue
        
        total_sources = supporting_sources + contradicting_sources
        if total_sources == 0:
            confidence = 0.5
        else:
            confidence = supporting_sources / (supporting_sources + contradicting_sources)
        
        return {
            "confidence": confidence,
            "supporting_sources": supporting_sources,
            "contradicting_sources": contradicting_sources
        }
    
    async def _detect_contradictions(self, results: List[ResearchResult]) -> List[Dict[str, str]]:
        """Detect contradictions between sources"""
        contradictions = []
        
        # Compare pairs of sources
        for i in range(len(results)):
            for j in range(i + 1, min(len(results), i + 4)):  # Limit comparisons
                source1, source2 = results[i], results[j]
                
                prompt = ChatPromptTemplate.from_template(
                    """Do these two sources contain contradictory information?
                    If yes, briefly describe the contradiction.
                    
                    Source 1: {content1}
                    
                    Source 2: {content2}
                    
                    Respond with: NO or describe the contradiction in one sentence."""
                )
                
                try:
                    response = await self.llm.ainvoke(
                        prompt.format_messages(
                            content1=source1.content[:600],
                            content2=source2.content[:600]
                        )
                    )
                    
                    if not response.content.strip().upper().startswith("NO"):
                        contradictions.append({
                            "source1": source1.source,
                            "source2": source2.source,
                            "contradiction": response.content.strip()
                        })
                        
                except:
                    continue
        
        return contradictions
    
    async def _assess_source_reliability(self, result: ResearchResult) -> float:
        """Assess the reliability of a source"""
        # Simple heuristic-based reliability assessment
        reliability_score = 0.5  # Base score
        
        source_url = result.source.lower()
        
        # High reliability domains
        if any(domain in source_url for domain in [
            'wikipedia.org', 'nature.com', 'science.org', 'ieee.org',
            'acm.org', 'springer.com', 'elsevier.com', 'pubmed.ncbi.nlm.nih.gov'
        ]):
            reliability_score += 0.3
        
        # Government and educational domains
        elif any(domain in source_url for domain in ['.gov', '.edu', '.org']):
            reliability_score += 0.2
        
        # Commercial domains (lower reliability)
        elif any(domain in source_url for domain in ['.com', '.net']):
            reliability_score += 0.1
        
        # Penalize for certain indicators
        if any(indicator in source_url for indicator in ['blog', 'forum', 'social']):
            reliability_score -= 0.1
        
        return max(0.0, min(1.0, reliability_score))

class ContentGenerationAgent(BaseAgent):
    """Agent responsible for generating various types of content"""
    
    def __init__(self, llm_provider: str = "openai"):
        super().__init__(AgentRole.CONTENT_GENERATOR, llm_provider)
    
    
    async def execute(self, input: dict) -> dict:
        """Execute content generation tasks"""
        state: PipelineState = input
        start_time = time.time()
        self.metrics.current_status = "generating_content"
        
        try:
            request = state["request"]
            research_results = state["research_results"]
            fact_check_results = state["fact_check_results"]
            if not request or not research_results or not fact_check_results:
                raise ValueError("Missing request, research results, or fact-check results in state")
            # Handle both dict and object formats
            if isinstance(research_results, dict):
                research_results = [ResearchResult(**r) for r in research_results.get("results", [])]
            if not isinstance(fact_check_results, dict):
                fact_check_results = {
                    "verified_facts": [],
                    "questionable_claims": [],
                    "contradictions": [],
                    "confidence_scores": {},
                    "source_reliability": {}
                }
            
            # Extract content types from request
            if isinstance(request, dict):
                content_types = request.get("content_types", ["summary"])
                logger.info(f"Generating content types: {content_types}")
            else:
                content_types = getattr(request, 'content_types', ["summary"])
                logger.info(f"Generating content types: {content_types}")
            
            generated_content = []
            
            for content_type in content_types:
                content = await self._generate_content(content_type, request, research_results, fact_check_results)
                if content:
                    generated_content.append(content)
            
            state["generated_content"] = generated_content
            state["current_step"] = "content_generation_completed"
            state["progress"] = 75
            
            self._update_metrics(True, time.time() - start_time)
            logger.info(f"Content generation completed with {len(generated_content)} pieces")
            self.metrics.current_status = "idle"
            
            logger.info(f"Generated {len(generated_content)} content pieces")
            
        except Exception as e:
            logger.error(f"Content Generation Agent failed: {str(e)}")
            state["errors"].append(f"Content generation failed: {str(e)}")
            self._update_metrics(False, time.time() - start_time)
            self.metrics.current_status = "error"
        
        return state
    
    async def _generate_content(
        self, content_type: str, 
        request: ResearchRequest, 
        research_results: List[ResearchResult],
        fact_check_results: Dict[str, Any]
    ) -> Optional[ContentOutput]:
        """Generate specific type of content"""
        
        # Prepare verified information
        verified_info = self._prepare_verified_info(research_results, fact_check_results)
        
        if content_type == "summary":
            return await self._generate_summary(request, verified_info)
        elif content_type == "report":
            return await self._generate_report(request, verified_info)
        elif content_type == "presentation":
            return await self._generate_presentation(request, verified_info)
        elif content_type == "blog_post":
            return await self._generate_blog_post(request, verified_info)
        elif content_type == "academic_paper":
            return await self._generate_academic_paper(request, verified_info)
        else:
            logger.warning(f"Unknown content type: {content_type}")
            return None
    
    def _prepare_verified_info(self, research_results: List[ResearchResult],fact_check_results: Dict[str, Any]) -> str:
        """Prepare verified information for content generation"""
        verified_content = []
        
        # If no research results, use general knowledge
        if not research_results:
            verified_content.append("No specific research data available. Using general knowledge base.")
            return "\n".join(verified_content)
        
        # Use high-reliability sources
        reliable_sources = [
            result for result in research_results
            if fact_check_results.get("source_reliability", {}).get(result.source, 0.5) > 0.7
        ]
        
        # Include verified facts
        verified_facts = fact_check_results.get("verified_facts", [])
        if verified_facts:
            verified_content.append("Verified Facts:")
            verified_content.extend(f"• {fact}" for fact in verified_facts)
        
        # Include content from reliable sources
        for result in reliable_sources[:5]:
            verified_content.append(f"\nFrom {result.title}:")
            verified_content.append(result.content[:800])
        
        # If no reliable sources, use all sources
        if not reliable_sources and research_results:
            verified_content.append("\nResearch Sources:")
            for result in research_results[:3]:
                verified_content.append(f"\nFrom {result.title}:")
                verified_content.append(result.content[:800])
        
        return "\n".join(verified_content)
    
    async def _generate_summary(self, request: ResearchRequest, verified_info: str) -> ContentOutput:
        """Generate a comprehensive summary"""
        # Extract topic and audience
        if isinstance(request, dict):
            topic = request.get("topic", "Unknown Topic")
            audience = request.get("target_audience", "general")
        else:
            topic = getattr(request, 'topic', "Unknown Topic")
            audience = getattr(request, 'target_audience', "general")
            
        prompt = ChatPromptTemplate.from_template(
            """Create a comprehensive summary about "{topic}" for a {audience} audience.
            
            Research Information:
            {verified_info}
            
            Requirements:
            - 300-500 words
            - Clear, engaging introduction
            - Key points with supporting evidence
            - Balanced perspective
            - Professional tone
            - Include implications and future outlook
            
            Summary:"""
        )
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                topic=topic,
                audience=audience,
                verified_info=verified_info[:4000] if verified_info else f"Topic: {topic}"
            )
        )
        
        content = response.content.strip()
        
        return ContentOutput(
            content_type="summary",
            title=f"Summary: {topic}",
            content=content,
            summary=content[:200] + "..." if len(content) > 200 else content,
            word_count=len(content.split()),
            readability_score=self._calculate_readability(content),
            sources_used=self._extract_sources(verified_info)
        )
    
    async def _generate_report(self, request: ResearchRequest, verified_info: str) -> ContentOutput:
        """Generate a detailed report"""
        prompt = ChatPromptTemplate.from_template(
            """Create a detailed research report about "{topic}" for a {audience} audience.
            
            Research Information:
            {verified_info}
            
            Structure the report with:
            1. Executive Summary
            2. Introduction and Background
            3. Key Findings
            4. Detailed Analysis
            5. Implications and Recommendations
            6. Conclusion
            
            Requirements:
            - 800-1200 words
            - Professional formatting
            - Data-driven insights
            - Clear section headers
            - Balanced analysis
            
            Report:"""
        )
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                topic=request.topic,
                audience=request.target_audience,
                verified_info=verified_info[:4000] if verified_info else f"Topic: {request.topic}"
            )
        )
        
        content = response.content.strip()
        
        return ContentOutput(
            content_type="report",
            title=f"Research Report: {request.topic}",
            content=content,
            summary=self._extract_executive_summary(content),
            word_count=len(content.split()),
            readability_score=self._calculate_readability(content),
            sources_used=self._extract_sources(verified_info)
        )
    
    async def _generate_presentation(self, request: ResearchRequest, verified_info: str) -> ContentOutput:
        """Generate presentation content"""
        prompt = ChatPromptTemplate.from_template(
            """Create presentation content about "{topic}" for a {audience} audience.
            
            Research Information:
            {verified_info}
            
            Create 8-12 slides with:
            - Title slide
            - Agenda/Overview
            - Key concept slides (3-5)
            - Data/statistics slides (2-3)
            - Implications slide
            - Conclusion slide
            - Q&A slide
            
            Format each slide as:
            SLIDE X: [Title]
            • Bullet point 1
            • Bullet point 2
            • Key insight or statistic
            
            Presentation:"""
        )
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                topic=request.topic,
                audience=request.target_audience,
                verified_info=verified_info[:4000] if verified_info else f"Topic: {request.topic}"
            )
        )
        
        content = response.content.strip()
        slide_lines = [l for l in content.split("\n") if l.startswith("SLIDE")]
        summary = f"Presentation covering {request.topic} with {len(slide_lines)} slides"

        
        return ContentOutput(
            content_type="presentation",
            title=f"Presentation: {request.topic}",
            content=content,
            summary=summary,
            word_count=len(content.split()),
            readability_score=self._calculate_readability(content),
            sources_used=self._extract_sources(verified_info)
        )
    
    async def _generate_blog_post(self, request: ResearchRequest, verified_info: str) -> ContentOutput:
        """Generate an engaging blog post"""
        prompt = ChatPromptTemplate.from_template(
            """Write an engaging blog post about "{topic}" for a {audience} audience.
            
            Research Information:
            {verified_info}
            
            Requirements:
            - Compelling headline
            - Engaging introduction with hook
            - 3-4 main sections with subheadings
            - Personal insights and examples
            - Call to action or thought-provoking conclusion
            - 600-800 words
            - Conversational yet informative tone
            
            Blog Post:"""
        )
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                topic=request.topic,
                audience=request.target_audience,
                verified_info=verified_info[:4000]
            )
        )
        
        content = response.content.strip()
        
        return ContentOutput(
            content_type="blog_post",
            title=f"Blog Post: {request.topic}",
            content=content,
            summary=content.split('\n\n')[1] if '\n\n' in content else content[:200] + "...",
            word_count=len(content.split()),
            readability_score=self._calculate_readability(content),
            sources_used=self._extract_sources(verified_info)
        )
    
    async def _generate_academic_paper(self, request: ResearchRequest, verified_info: str) -> ContentOutput:
        """Generate academic paper outline"""
        prompt = ChatPromptTemplate.from_template(
            """Create an academic paper outline about "{topic}".
            
            Research Information:
            {verified_info}
            
            Structure:
            1. Abstract (150-200 words)
            2. Introduction
            3. Literature Review
            4. Methodology (if applicable)
            5. Findings/Results
            6. Discussion
            7. Conclusion
            8. References (placeholder)
            
            Requirements:
            - Academic tone and style
            - Proper section headers
            - Key points for each section
            - Potential research questions
            - Scholarly approach
            
            Academic Paper Outline:"""
        )
        
        response = await self.llm.ainvoke(prompt.format_messages(topic=request.topic, verified_info=verified_info[:4000]))
        
        content = response.content.strip()
        
        return ContentOutput(
            content_type="academic_paper",
            title=f"Academic Paper Outline: {request.topic}",
            content=content,
            summary=self._extract_abstract(content),
            word_count=len(content.split()),
            readability_score=self._calculate_readability(content),
            sources_used=self._extract_sources(verified_info)
        )
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score (0-1)"""
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = words / sentences
        
        # Simple heuristic: shorter sentences = higher readability
        if avg_words_per_sentence < 15:
            return 0.8
        elif avg_words_per_sentence < 25:
            return 0.6
        else:
            return 0.4
    
    def _extract_sources(self, verified_info: str) -> List[str]:
        """Extract source references from verified info"""
        sources = []
        lines = verified_info.split('\n')
        
        for line in lines:
            if line.startswith('From '):
                source = line.replace('From ', '').replace(':', '').strip()
                if source:
                    sources.append(source)
        
        return sources
    
    def _extract_executive_summary(self, content: str) -> str:
        """Extract executive summary from report"""
        lines = content.split('\n')
        summary_lines = []
        in_summary = False
        
        for line in lines:
            if 'executive summary' in line.lower():
                in_summary = True
                continue
            if in_summary and line.strip() and not line.startswith('#'):
                summary_lines.append(line.strip())
            elif in_summary and (line.startswith('#') or 'introduction' in line.lower()):
                break
        
        return ' '.join(summary_lines[:3]) if summary_lines else content[:200] + "..."
    
    def _extract_abstract(self, content: str) -> str:
        """Extract abstract from academic paper"""
        lines = content.split('\n')
        abstract_lines = []
        in_abstract = False
        
        for line in lines:
            if 'abstract' in line.lower() and not in_abstract:
                in_abstract = True
                continue
            if in_abstract and line.strip() and not line.startswith('#'):
                abstract_lines.append(line.strip())
            elif in_abstract and (line.startswith('#') or 'introduction' in line.lower()):
                break
        
        return ' '.join(abstract_lines) if abstract_lines else content[:200] + "..."

class QualityAssuranceAgent(BaseAgent):
    """Agent responsible for quality control and final review"""
    
    def __init__(self, llm_provider: str = "openai"):
        super().__init__(AgentRole.QUALITY_ASSURANCE, llm_provider)
    
    
    async def execute(self, input: dict) -> dict:
        """Execute quality assurance tasks"""
        state: PipelineState = input
        start_time = time.time()
        self.metrics.current_status = "quality_checking"
        
        try:
            generated_content = state["generated_content"]
            fact_check_results = state["fact_check_results"]
            if not generated_content or not fact_check_results:
                raise ValueError("Missing generated content or fact-check results in state")
            # Handle both dict and object formats
            if isinstance(generated_content, dict):
                generated_content = [ContentOutput(**c) for c in generated_content.get("content", [])]
            elif isinstance(generated_content, list) and all(isinstance(c, dict) for c in generated_content):
                generated_content = [ContentOutput(**c) for c in generated_content]
            
            logger.info(f"Quality checking {len(generated_content)} content pieces")
            
            qa_results = {
                "overall_quality_score": 0.0,
                "content_scores": {},
                "issues_found": [],
                "recommendations": [],
                "approved_content": []
            }
            
            total_score = 0.0
            
            for content in generated_content:
                score = await self._assess_content_quality(content, fact_check_results)
                qa_results["content_scores"][content.content_type] = score
                total_score += score
                
                # Check for issues
                issues = await self._identify_issues(content, fact_check_results)
                qa_results["issues_found"].extend(issues)
                
                # Generate recommendations
                recommendations = await self._generate_recommendations(content, score, issues)
                qa_results["recommendations"].extend(recommendations)
                
                # Approve content above threshold
                if score > 0.7:
                    qa_results["approved_content"].append(content.content_type)
            
            qa_results["overall_quality_score"] = total_score / len(generated_content) if generated_content else 0.0
            
            state["agent_outputs"]["quality_assurance"] = qa_results
            state["current_step"] = "quality_assurance_completed"
            state["progress"] = 95
            
            self._update_metrics(True, time.time() - start_time)
            logger.info(f"Quality assurance completed with overall score: {qa_results['overall_quality_score']:.2f}")
            state["qa_results"] = qa_results
            state["errors"] = []
            self.metrics.current_status = "idle"
            
            logger.info(f"Quality assurance completed. Overall score: {qa_results['overall_quality_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Quality Assurance Agent failed: {str(e)}")
            state["errors"].append(f"Quality assurance failed: {str(e)}")
            self._update_metrics(False, time.time() - start_time)
            self.metrics.current_status = "error"
        
        return state
    
    async def _assess_content_quality(self, content: ContentOutput, fact_check_results: Dict[str, Any]) -> float:
        """Assess overall content quality"""
        prompt = ChatPromptTemplate.from_template(
            """Assess the quality of this {content_type} on a scale of 0.0 to 1.0.
            
            Title: {title}
            Content: {content}
            
            Evaluate:
            - Clarity and coherence
            - Factual accuracy
            - Completeness
            - Professional quality
            - Engagement level
            - Structure and organization
            
            Return only a number between 0.0 and 1.0"""
        )
        
        try:
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    content_type=content.content_type,
                    title=content.title,
                    content=content.content[:2000]
                )
            )
            
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
            
        except:
            return 0.6  # Default score if assessment fails
    
    async def _identify_issues(
        self, 
        content: ContentOutput, 
        fact_check_results: Dict[str, Any]
    ) -> List[str]:
        """Identify potential issues in content"""
        issues = []
        
        # Check for questionable claims
        questionable_claims = fact_check_results.get("questionable_claims", [])
        for claim in questionable_claims:
            if claim.lower() in content.content.lower():
                issues.append(f"Contains questionable claim: {claim[:100]}...")
        
        # Check for contradictions
        contradictions = fact_check_results.get("contradictions", [])
        if contradictions:
            issues.append(f"Source contradictions detected: {len(contradictions)} found")
        
        # Check content length
        word_count = content.word_count
        if content.content_type == "summary" and word_count > 600:
            issues.append("Summary too long - should be under 500 words")
        elif content.content_type == "report" and word_count < 600:
            issues.append("Report too short - should be 800+ words")
        
        # Check readability
        if content.readability_score < 0.4:
            issues.append("Low readability score - content may be too complex")
        
        return issues
    
    async def _generate_recommendations(
        self, 
        content: ContentOutput, 
        quality_score: float, 
        issues: List[str]
    ) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append(f"Consider regenerating {content.content_type} - quality score below threshold")
        
        if issues:
            recommendations.append(f"Address {len(issues)} identified issues in {content.content_type}")
        
        if content.readability_score < 0.5:
            recommendations.append(f"Improve readability of {content.content_type}")
        
        if not content.sources_used:
            recommendations.append(f"Add source citations to {content.content_type}")
        
        return recommendations

class ResearchPipeline:
    """Main pipeline orchestrator using LangGraph"""
    
    def __init__(self, llm_provider: str = "openai"):
        self.llm_provider = llm_provider
        self.research_agent = ResearchAgent(llm_provider)
        self.factcheck_agent = FactCheckingAgent(llm_provider)
        self.content_agent = ContentGenerationAgent(llm_provider)
        self.qa_agent = QualityAssuranceAgent(llm_provider)
        logger.info("Pipeline initialized with structured agent execution")
    
    async def initialize(self):
        """Initialize the pipeline and test agent connectivity"""
        logger.info("Initializing ResearchPipeline")
        for role, agent in [
            (AgentRole.RESEARCHER, self.research_agent),
            (AgentRole.FACT_CHECKER, self.factcheck_agent),
            (AgentRole.CONTENT_GENERATOR, self.content_agent),
            (AgentRole.QUALITY_ASSURANCE, self.qa_agent),
        ]:
            try:
                # Simple ping to the LLM
                await agent.llm.ainvoke([HumanMessage(content="Hello")])
                logger.info(f"✓ {role.value} agent initialized")
            except Exception as e:
                logger.warning(f"⚠ {role.value} initialization failed: {e}")
    
    async def execute_pipeline(self, pipeline_id: str, request: ResearchRequest, progress_callback: Optional[Callable[[str, int, str], None]] = None) -> Dict[str, Any]:
        """
        Execute the complete research pipeline with all agents in sequence
        """
        execution_start = datetime.now()
        
        # Initialize pipeline state
        pipeline_state: PipelineState = {
            "request": request,
            "research_results": [],
            "fact_check_results": {},
            "generated_content": [],
            "current_step": "initializing",
            "progress": 0,
            "errors": [],
            "agent_outputs": {},
            "execution_start": execution_start,
            "metadata": {"pipeline_id": pipeline_id}
        }
        
        try:
            logger.info(f"Starting pipeline execution for: {request.topic}")
            
            # Update pipeline status in fastapi
            if progress_callback:
                progress_callback("initializing", 0, "Pipeline starting...")
            
            # Step 1: Research Agent
            logger.info("Step 1: Executing Research Agent")
            pipeline_state["current_step"] = "research"
            pipeline_state["progress"] = 10
            
            if progress_callback:
                progress_callback("research", 10, "Gathering research data...")
            
            pipeline_state = await self.research_agent.execute(pipeline_state)
            
            if pipeline_state["errors"]:
                logger.warning(f"Research agent completed with errors: {pipeline_state['errors']}")
            
            logger.info(f"Research completed with {len(pipeline_state['research_results'])} results")
            
            # Step 2: Fact checking Agent
            logger.info("Step 2: Executing Fact-Checking Agent")
            pipeline_state["current_step"] = "fact_checking"
            pipeline_state["progress"] = 40
            
            if progress_callback:
                progress_callback("fact_checking", 40, "Verifying information accuracy...")
            
            pipeline_state = await self.factcheck_agent.execute(pipeline_state)
            
            if pipeline_state["errors"]:
                logger.warning(f"Fact-checking completed with errors: {pipeline_state['errors']}")
            
            logger.info("Fact-checking completed")
            
            # Step 3: Content Generation Agent
            logger.info("Step 3: Executing Content Generation Agent")
            pipeline_state["current_step"] = "content_generation"
            pipeline_state["progress"] = 70
            
            if progress_callback:
                progress_callback("content_generation", 70, "Generating content...")
            
            pipeline_state = await self.content_agent.execute(pipeline_state)
            
            if pipeline_state["errors"]:
                logger.warning(f"Content generation completed with errors: {pipeline_state['errors']}")
            
            logger.info(f"Content generation completed with {len(pipeline_state['generated_content'])} pieces")
            
            # Step 4: Quality Assurance Agent
            logger.info("Step 4: Executing Quality Assurance Agent")
            pipeline_state["current_step"] = "quality_assurance"
            pipeline_state["progress"] = 90
            
            if progress_callback:
                progress_callback("quality_assurance", 90, "Performing quality checks...")
            
            pipeline_state = await self.qa_agent.execute(pipeline_state)
            
            if pipeline_state["errors"]:
                logger.warning(f"Quality assurance completed with errors: {pipeline_state['errors']}")
            
            # Final completion
            pipeline_state["current_step"] = "completed"
            pipeline_state["progress"] = 100
            
            if progress_callback:
                progress_callback("completed", 100, "Pipeline completed successfully")
            
            # Prepare final results
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            final_results = {
                "pipeline_id": pipeline_id,
                "status": "completed",
                "execution_time": execution_time,
                "request": request.dict() if hasattr(request, 'dict') else request,
                "research_summary": {
                    "total_sources": len(pipeline_state["research_results"]),
                    "sources": [
                        {
                            "title": result.title if hasattr(result, 'title') else result.get('title', 'Unknown'),
                            "source": result.source if hasattr(result, 'source') else result.get('source', 'Unknown'),
                            "relevance_score": result.relevance_score if hasattr(result, 'relevance_score') else result.get('relevance_score', 0.0)
                        }
                        for result in pipeline_state["research_results"]
                    ]
                },
                "fact_check_summary": {
                    "verified_facts": len(pipeline_state["fact_check_results"].get("verified_facts", [])),
                    "questionable_claims": len(pipeline_state["fact_check_results"].get("questionable_claims", [])),
                    "contradictions": len(pipeline_state["fact_check_results"].get("contradictions", [])),
                    "overall_reliability": sum(pipeline_state["fact_check_results"].get("source_reliability", {}).values()) / 
                                         max(len(pipeline_state["fact_check_results"].get("source_reliability", {})), 1)
                },
                "generated_content": [
                    {
                        "content_type": content.content_type if hasattr(content, 'content_type') else content.get('content_type', 'unknown'),
                        "title": content.title if hasattr(content, 'title') else content.get('title', 'Untitled'),
                        "word_count": content.word_count if hasattr(content, 'word_count') else content.get('word_count', 0),
                        "readability_score": content.readability_score if hasattr(content, 'readability_score') else content.get('readability_score', 0.0),
                        "content": content.content if hasattr(content, 'content') else content.get('content', ''),
                        "summary": content.summary if hasattr(content, 'summary') else content.get('summary', ''),
                        "sources_used": content.sources_used if hasattr(content, 'sources_used') else content.get('sources_used', [])
                    }
                    for content in pipeline_state["generated_content"]
                ],
                "quality_assessment": pipeline_state["agent_outputs"].get("quality_assurance", {}),
                "errors": pipeline_state["errors"],
                "agent_metrics": self.get_agent_metrics()
            }
            
            logger.info(f"Pipeline completed successfully in {execution_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            
            # Update status to failed
            if progress_callback:
                progress_callback("failed", pipeline_state.get("progress", 0), f"Pipeline failed: {str(e)}")
            
            # Return error result
            execution_time = (datetime.now() - execution_start).total_seconds()
            
            return {
                "pipeline_id": pipeline_id,
                "status": "failed",
                "execution_time": execution_time,
                "request": request.dict() if hasattr(request, 'dict') else request,
                "error": str(e),
                "errors": pipeline_state.get("errors", []),
                "current_step": pipeline_state.get("current_step", "unknown"),
                "progress": pipeline_state.get("progress", 0),
                "partial_results": {
                    "research_results": len(pipeline_state.get("research_results", [])),
                    "fact_check_completed": bool(pipeline_state.get("fact_check_results")),
                    "content_generated": len(pipeline_state.get("generated_content", []))
                }
            }
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        logger.info("Cleaning up pipeline resources")
        for agent in [self.research_agent, self.factcheck_agent, self.content_agent, self.qa_agent]:
            if hasattr(agent.metrics, 'reset'):
                agent.metrics.reset()  # Reset metrics for next run
            logger.info(f"✓ {agent.role.value} agent cleaned up")

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current status of all agents"""
        return {
            role.value: {
                "status": agent.metrics.current_status,
                "total_executions": agent.metrics.total_executions,
                "success_rate": agent.metrics.successful_executions / agent.metrics.total_executions if agent.metrics.total_executions > 0 else 0.0,
                "avg_exec_time": agent.metrics.avg_execution_time,
                "last_exec": agent.metrics.last_execution.isoformat() if agent.metrics.last_execution else None
            }
            for role, agent in [
                (AgentRole.RESEARCHER, self.research_agent),
                (AgentRole.FACT_CHECKER, self.factcheck_agent),
                (AgentRole.CONTENT_GENERATOR, self.content_agent),
                (AgentRole.QUALITY_ASSURANCE, self.qa_agent),
            ]
        }

    def get_agent_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for all agents"""
        return {
            role.value: agent.metrics.dict() if hasattr(agent.metrics, 'dict') else {
                "agent_id": agent.metrics.agent_id,
                "role": agent.metrics.role,
                "total_executions": agent.metrics.total_executions,
                "successful_executions": agent.metrics.successful_executions,
                "failed_executions": agent.metrics.failed_executions,
                "avg_execution_time": agent.metrics.avg_execution_time,
                "current_status": agent.metrics.current_status,
                "last_execution": agent.metrics.last_execution.isoformat() if agent.metrics.last_execution else None
            }
            for role, agent in [
                (AgentRole.RESEARCHER, self.research_agent),
                (AgentRole.FACT_CHECKER, self.factcheck_agent),
                (AgentRole.CONTENT_GENERATOR, self.content_agent),
                (AgentRole.QUALITY_ASSURANCE, self.qa_agent),
            ]
        }