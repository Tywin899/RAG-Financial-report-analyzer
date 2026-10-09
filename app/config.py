import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    """Application settings and configuration."""
    app_title: str = "Financial Report RAG API"
    app_version: str = "2.0.0"
    
    # Model Configurations
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_model_name: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL_NAME", os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")))
    llm_temperature: float = 0.0
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    # Document Processing Configurations
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4
    
    # Upload limits
    max_file_size_bytes: int = 25 * 1024 * 1024  # 25MB
    allowed_mime_types: List[str] = ["application/pdf"]
    allowed_extensions: List[str] = [".pdf"]


settings = Settings()
