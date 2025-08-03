"""
Configuration management for the intelligent routing system
"""

import os
from typing import Dict, Any
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Redis Settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    
    # ChromaDB Settings
    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "research_results"
    
    # Cache Settings
    cache_ttl_hours: int = 48
    similarity_threshold: float = 0.80
    cache_threshold: float = 0.90
    rag_threshold: float = 0.80
    
    # RAG Settings
    rag_top_k: int = 3
    rag_confidence_threshold: float = 0.7
    
    # LLM Settings
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    anthropic_api_key: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()