from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime

class PipelineRequest(BaseModel):
    """Request model for pipeline execution"""
    topic: str = Field(..., description="Research topic", min_length=3, max_length=200)
    depth: str = Field(default="standard", description="Research depth: basic, standard, deep")
    content_types: List[str] = Field(default=["summary", "report"], description="Content types to generate")
    target_audience: str = Field(default="general", description="Target audience level")
    max_sources: int = Field(default=10, ge=3, le=50, description="Maximum sources to research")

class PipelineResponse(BaseModel):
    """Response model for pipeline status"""
    pipeline_id: str
    status: str
    message: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None

class ResearchRequest(BaseModel):
    """Research request configuration"""
    topic: str = Field(..., description="Main research topic")
    depth: str = Field(default="standard", description="Research depth level")
    content_types: List[str] = Field(default=["summary", "report"])
    target_audience: str = Field(default="general")
    max_sources: int = Field(default=10, ge=1, le=50)
    language: str = Field(default="en")
    include_citations: bool = Field(default=True)
    
    @field_validator('depth')
    def validate_depth(cls, v):
        allowed = ["basic", "standard", "deep", "comprehensive"]
        if v not in allowed:
            raise ValueError(f"Depth must be one of {allowed}")
        return v

class AgentRole(str, Enum):
    """Agent roles in the system"""
    RESEARCHER = "researcher"
    FACT_CHECKER = "fact_checker"
    CONTENT_GENERATOR = "content_generator"
    QUALITY_ASSURANCE = "quality_assurance"
    COORDINATOR = "coordinator"

class AgentMetrics(BaseModel):
    """Agent performance metrics"""
    agent_id: str
    role: AgentRole
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    last_execution: Optional[datetime] = None
    current_status: str = "idle"

class ResearchResult(BaseModel):
    """Single research result"""
    title: str
    content: str
    source: str
    credibility_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)

class ContentOutput(BaseModel):
    """Generated content output"""
    content_type: str
    title: str
    content: str
    summary: str
    word_count: int
    readability_score: float
    sources_used: List[str]
    generated_at: datetime = Field(default_factory=datetime.now)

class RouteType(str, Enum):
    """Routing decision types"""
    CACHE = "cache"
    RAG = "rag" 
    PIPELINE = "pipeline"

class CacheEntry(BaseModel):
    """Cache entry model"""
    query: str
    chroma_id: str
    content: List[Dict[str, Any]]
    cached_at: datetime
    ttl_seconds: int
    source: str = "pipeline"

class SimilarQuery(BaseModel):
    """Similar query model"""
    query: str
    similarity: float = Field(ge=0.0, le=1.0)
    cached_at: Optional[str] = None
    chroma_id: Optional[str] = None

class RAGResponse(BaseModel):
    """RAG response model"""
    generated_content: List[ContentOutput]
    retrieved_docs: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time: Optional[float] = None

class RoutingDecision(BaseModel):
    """Routing decision model"""
    route_type: RouteType
    similarity_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    similar_queries: List[SimilarQuery] = []

class IntelligentResponse(BaseModel):
    """Enhanced response with routing information"""
    pipeline_id: str
    status: str
    source: RouteType
    execution_time: float
    routing_decision: RoutingDecision
    generated_content: List[ContentOutput]
    cache_hit: bool = False
    rag_used: bool = False
    metadata: Dict[str, Any] = {}

class CacheStats(BaseModel):
    """Cache statistics model"""
    redis_keys: int
    chromadb_documents: int
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    avg_similarity_score: float = Field(ge=0.0, le=1.0)
    last_cleanup: Optional[datetime] = None

class WorkflowMetrics(BaseModel):
    """Workflow performance metrics"""
    total_queries: int = 0
    cache_hits: int = 0
    rag_hits: int = 0
    pipeline_runs: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    efficiency_score: float = Field(ge=0.0, le=1.0, default=0.0)