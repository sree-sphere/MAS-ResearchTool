"""
Multi-Agent Research & Content Pipeline
A sophisticated multi-agent system for automated research, fact-checking, and content generation.

Author: sree-sphere
Description: Enterprise-grade multi-agent system using LangGraph for orchestrating
            research, analysis, and content creation workflows.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from src.multi_agent_system import (
    ResearchPipeline,
    ResearchRequest,
    PipelineStatus,
    AgentMetrics,
    ContentOutput
)

# Logger config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/research_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API init
app = FastAPI(
    title="Multi-Agent Research Pipeline",
    description="Automated research and content generation using AI agents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
research_pipeline = ResearchPipeline()

# In-memory storage for demo
pipeline_results: Dict[str, Dict] = {}
active_pipelines: Dict[str, bool] = {}

# Pydantic models for request/response
class PipelineRequest(BaseModel):
    """Request model for pipeline execution"""
    topic: str = Field(..., description="Research topic", min_length=3, max_length=200)
    depth: str = Field(default="standard", description="Research depth: basic, standard, deep")
    content_types: List[str] = Field(
        default=["summary", "report"], 
        description="Content types to generate"
    )
    target_audience: str = Field(default="general", description="Target audience level")
    max_sources: int = Field(default=10, ge=3, le=50, description="Maximum sources to research")

class PipelineResponse(BaseModel):
    """Response model for pipeline status"""
    pipeline_id: str
    status: str
    message: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Multi Agent Systen (MAS) => Research Pipeline",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "agents": research_pipeline.get_agent_status()
    }

@app.post("/research/start", response_model=PipelineResponse)
async def start_research_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks
):
    """Start a new research pipeline"""
    try:
        # 1. Generate unique pipeline ID
        pipeline_id = (
            "pipeline_" +
            datetime.now().strftime("%Y%m%d_%H%M%S") +
            f"_{hash(request.topic) % 10000}"
        )
        
        # 2. Prepare the research request object
        research_request = ResearchRequest(
            topic=request.topic,
            depth=request.depth,
            content_types=request.content_types,
            target_audience=request.target_audience,
            max_sources=request.max_sources
        )
        
        # 3. Initialize pipeline tracking
        now = datetime.utcnow()
        active_pipelines[pipeline_id] = True
        pipeline_results[pipeline_id] = {
            "status": "initializing",
            "request": request.dict(),
            "created_at": now,
            "progress": 0,
            "current_agent": None,
            "results": None,
            "error": None
        }
        
        # 4. Kick off the background pipeline
        background_tasks.add_task(
            execute_pipeline,
            pipeline_id,
            research_request
        )
        
        logger.info(f"Started research pipeline {pipeline_id} for topic: {request.topic}")
        
        # 5. Respond with estimated completion
        return PipelineResponse(
            pipeline_id=pipeline_id,
            status="started",
            message=f"Research pipeline initiated for topic: {request.topic}",
            created_at=now,
            estimated_completion=now + timedelta(minutes=3)             # using timedelta rather than replace() to avoid timezone issues
        )
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")

@app.get("/research/status/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """Get pipeline execution status"""
    if pipeline_id not in pipeline_results:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    result = pipeline_results[pipeline_id]
    
    return {
        "pipeline_id": pipeline_id,
        "status": result["status"],
        "progress": result["progress"],
        "current_agent": result["current_agent"],
        "created_at": result["created_at"].isoformat(),
        "has_results": result["results"] is not None,
        "error": result["error"]
    }

@app.get("/research/results/{pipeline_id}")
async def get_pipeline_results(pipeline_id: str):
    """Get pipeline results"""
    if pipeline_id not in pipeline_results:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    result = pipeline_results[pipeline_id]
    
    if result["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=202, detail="Pipeline still running")
    
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result["error"])
    
    if not result["results"]:
        raise HTTPException(status_code=404, detail="No results found for this pipeline")
    
    if not isinstance(result["results"], ContentOutput):
        raise HTTPException(status_code=500, detail="Invalid results format")
    
    # Response Payload
    payload = {
        "pipeline_id": pipeline_id,
        "status": result["status"],
        "results": result["results"],
        "completed_at": result.get("completed_at"),
        "execution_time": result.get("execution_time"),
        "agent_metrics": result.get("agent_metrics"),
    }
    # Convert any datetime/UUID/etc → JSON primitives
    safe_payload = jsonable_encoder(payload)
    return JSONResponse(content=safe_payload)

@app.get("/research/active")
async def get_active_pipelines():
    """Get all active pipelines"""
    active = [
        {
            "pipeline_id": pid,
            "status": pipeline_results[pid]["status"],
            "progress": pipeline_results[pid]["progress"],
            "topic": pipeline_results[pid]["request"]["topic"],
            "created_at": pipeline_results[pid]["created_at"].isoformat()
        }
        for pid in pipeline_results
        if pipeline_results[pid]["status"] not in ["completed", "failed"]
    ]
    
    return {"active_pipelines": active, "count": len(active)}

@app.delete("/research/cancel/{pipeline_id}")
async def cancel_pipeline(pipeline_id: str):
    """Cancel a running pipeline"""
    if pipeline_id not in active_pipelines:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    
    active_pipelines[pipeline_id] = False
    if pipeline_id in pipeline_results:
        pipeline_results[pipeline_id]["status"] = "cancelled"
    
    return {"message": f"Pipeline {pipeline_id} cancelled"}

@app.get("/agents/status")
async def get_agent_status():
    """Get status of all agents"""
    return research_pipeline.get_agent_status()

@app.get("/agents/metrics")
async def get_agent_metrics():
    """Get agent performance metrics"""
    return research_pipeline.get_agent_metrics()

# Execution task
async def execute_pipeline(pipeline_id: str, request: ResearchRequest):
    """Execute research pipeline in background"""
    try:
        logger.info(f"Executing pipeline {pipeline_id}")
        
        # Set pipeline status
        pipeline_results[pipeline_id]["status"] = "running"
        pipeline_results[pipeline_id]["progress"] = 5
        
        # Execute pipeline with progress tracking
        start_time = datetime.now()
        
        async def progress_callback(agent_name: str, progress: int, message: str):
            if pipeline_id in active_pipelines and active_pipelines[pipeline_id]:
                pipeline_results[pipeline_id]["progress"] = progress
                pipeline_results[pipeline_id]["current_agent"] = agent_name
                logger.info(f"Pipeline {pipeline_id}: {agent_name} - {message} ({progress}%)")
            else:
                # Pipeline cancelled
                raise Exception("Pipeline cancelled by user")
        
        # Run research pipeline
        results = await research_pipeline.execute_pipeline(request, progress_callback)
        
        end_time = datetime.now()
        
        # Update final results
        pipeline_results[pipeline_id].update({
            "status": "completed",
            "progress": 100,
            "current_agent": "completed",
            "results": results.dict() if results else None,
            "completed_at": end_time,
            "execution_time": (end_time - start_time).total_seconds(),
            "agent_metrics": research_pipeline.get_agent_metrics()
        })
        
        # Save final results
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        output_path = Path("outputs") / f"{pipeline_id}_results.json"
        with open(output_path, "w") as f:
            json.dump(pipeline_results[pipeline_id], f, indent=2)
        
        logger.info(f"Pipeline {pipeline_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline {pipeline_id} failed: {str(e)}")
        pipeline_results[pipeline_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now()
        })
        
        # Save error state
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        output_path = Path("outputs") / f"{pipeline_id}_error.json"
        with open(output_path, "w") as f:
            json.dump(pipeline_results[pipeline_id], f, indent=2)
    
    finally:
        # Cleanup
        if pipeline_id in active_pipelines:
            del active_pipelines[pipeline_id]

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("Starting Multi-Agent Research Pipeline")
    
    # Create output directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Pipeline init
    await research_pipeline.initialize()
    
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Multi-Agent Research Pipeline")
    
    # Cancel all active pipelines
    for pipeline_id in list(active_pipelines.keys()):
        active_pipelines[pipeline_id] = False
    
    # Cleanup pipeline resources
    await research_pipeline.cleanup()
    
    logger.info("Application shutdown complete")

def main():
    print("Multi-Agent Research Pipeline")
    print("=" * 50)
    print("Features:")
    print("  • Intelligent Research Agent")
    print("  • Fact-Checking Agent")
    print("  • Content Generation Agent")
    print("  • Quality Assurance Agent")
    print("  • RESTful API Interface")
    print("  • Real-time Progress Tracking")
    print("  • Comprehensive Logging")
    print("=" * 50)
    
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: No API key found. Set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    print("\nStarting server...")
    print("   API Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/")
    print("\n   Example curl command:")
    print('   curl -X POST "http://localhost:8000/research/start" \\')
    print('        -H "Content-Type: application/json" \\')
    print('        -d \'{"topic": "Artificial Intelligence in Healthcare", "depth": "standard"}\'')
    
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

if __name__ == "__main__":
    main()