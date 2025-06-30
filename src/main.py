"""
Multi-Agent Research & Content Pipeline

Author: sree-sphere
Description: Multi-agent system (MAS) using LangGraph for orchestrating research, analysis, and content creation workflows.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from uuid import uuid4
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from dotenv import load_dotenv
from pydantic import ValidationError

from src.log import logger
from src.models.models import PipelineRequest, PipelineResponse, ResearchRequest, ContentOutput
from src.multi_agent_system import ResearchPipeline

load_dotenv()

# App init
app = FastAPI(
    title="Multi-Agent Research Pipeline",
    description="Automated research and content generation using AI agents",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
research_pipeline = ResearchPipeline()
pipeline_results: Dict[str, Dict] = {}
active_pipelines: Dict[str, bool] = {}


async def execute_pipeline_with_callback(pipeline_id: str, research_request: ResearchRequest):
    """Run pipeline and update progress/result"""
    try:
        def progress_callback(step: str, progress: int, message: str):
            if pipeline_id in pipeline_results:
                pipeline_results[pipeline_id].update({
                    "current_step": step,
                    "progress": progress,
                    "status": step,
                    "message": message
                })

        results = await research_pipeline.execute_pipeline(pipeline_id, research_request, progress_callback)

        if pipeline_id in pipeline_results:
            end_time = datetime.utcnow()
            pipeline_results[pipeline_id].update({
                "status": results["status"],
                "results": results,
                "progress": 100,
                "current_step": "completed",
                "completed_at": end_time,
                "execution_time": (end_time - pipeline_results[pipeline_id]["created_at"]).total_seconds(),
                "agent_metrics": research_pipeline.get_agent_metrics()
            })

            output_path = Path("outputs") / f"{pipeline_id}_results.json"
            with open(output_path, "w") as f:
                json.dump(pipeline_results[pipeline_id], f, indent=2, default=str)

    except Exception as e:
        logger.error(f"Pipeline {pipeline_id} failed: {str(e)}")
        if pipeline_id in pipeline_results:
            pipeline_results[pipeline_id].update({
                "status": "failed",
                "error": str(e)
            })
            output_path = Path("outputs") / f"{pipeline_id}_error.json"
            with open(output_path, "w") as f:
                json.dump(pipeline_results[pipeline_id], f, indent=2, default=str)
    finally:
        active_pipelines.pop(pipeline_id, None)


@app.post("/research/start", response_model=PipelineResponse)
async def start_research_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Start a new research pipeline"""
    try:
        pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        if pipeline_id in active_pipelines:
            raise HTTPException(status_code=400, detail="Pipeline already exists with this ID")

        research_request = ResearchRequest(**request.dict())
        now = datetime.utcnow()
        active_pipelines[pipeline_id] = True
        pipeline_results[pipeline_id] = {
            "status": "initializing",
            "request": request.dict(),
            "created_at": now,
            "progress": 0,
            "current_step": None,
            "results": None,
            "error": None
        }

        background_tasks.add_task(
            execute_pipeline_with_callback,
            pipeline_id,
            research_request
        )

        logger.info(f"Started research pipeline {pipeline_id} for topic: {request.topic}")

        return PipelineResponse(
            pipeline_id=pipeline_id,
            status="started",
            message=f"Research pipeline initiated for topic: {request.topic}",
            created_at=now,
            estimated_completion=now + timedelta(minutes=3)
        )
    except Exception as e:
        logger.error(f"Failed to start pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")

@app.get("/research/status/{pipeline_id}")
async def get_pipeline_status(pipeline_id: str):
    """Check pipeline execution status"""
    if pipeline_id not in pipeline_results:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    result = pipeline_results[pipeline_id]
    return {
        "pipeline_id": pipeline_id,
        "status": result["status"],
        "progress": result["progress"],
        "current_step": result["current_step"],
        "created_at": result["created_at"].isoformat(),
        "has_results": result["results"] is not None,
        "error": result["error"]
    }

@app.get("/research/results/{pipeline_id}")
async def get_pipeline_results(pipeline_id: str):
    """Fetch completed results"""
    if pipeline_id not in pipeline_results:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    result = pipeline_results[pipeline_id]

    if result["status"] not in ["completed", "failed"]:
        raise HTTPException(status_code=202, detail="Pipeline still running")

    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))

    if not result.get("results"):
        raise HTTPException(status_code=404, detail="No results found for this pipeline")

    try:
        validated_results = [ContentOutput.model_validate(item) for item in result["results"]["generated_content"]]

    except ValidationError:
        raise HTTPException(status_code=500, detail="Invalid results format")

    payload = {
        "pipeline_id": pipeline_id,
        "status": result["status"],
        "results": validated_results,
        "completed_at": result.get("completed_at"),
        "execution_time": result.get("execution_time"),
        "agent_metrics": result.get("agent_metrics"),
    }

    return JSONResponse(content=jsonable_encoder(payload))


@app.get("/research/active")
async def get_active_pipelines():
    """List active pipelines"""
    active = [
        {
            "pipeline_id": pid,
            "status": pipeline_results[pid]["status"],
            "progress": pipeline_results[pid]["progress"],
            "topic": pipeline_results[pid]["request"]["topic"],
            "created_at": pipeline_results[pid]["created_at"].isoformat()
        }
        for pid in pipeline_results
        if pipeline_results[pid]["status"] not in ["completed", "failed", "cancelled"]
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
    return research_pipeline.get_agent_status()

@app.get("/agents/metrics")
async def get_agent_metrics():
    return research_pipeline.get_agent_metrics()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Multi-Agent Research Pipeline")
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    await research_pipeline.initialize()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Multi-Agent Research Pipeline")
    for pipeline_id in list(active_pipelines.keys()):
        active_pipelines[pipeline_id] = False
    await research_pipeline.cleanup()
    logger.info("Application shutdown complete")

def main():
    verbose = "--dev" in sys.argv or "--verbose" in sys.argv

    if verbose:
        print("Multi-Agent Research Pipeline")
        print("=" * 50)
        print("  • Intelligent Research Agent")
        print("  • Fact-Checking Agent")
        print("  • Content Generation Agent")
        print("  • Quality Assurance Agent")
        print("  • RESTful API Interface")
        print("  • Real-time Progress Tracking")
        print("  • Logging & Metrics")
        print("=" * 50)

    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: No API key found. Set either OPENAI_API_KEY or ANTHROPIC_API_KEY")

    print("Starting server..." if verbose else "")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")

if __name__ == "__main__":
    main()