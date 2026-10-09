"""Request and response schemas."""
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    """Schema for asking questions about the uploaded report."""
    question: str = Field(..., description="Financial question to ask regarding the uploaded report.")

    @field_validator("question")
    @classmethod
    def validate_non_empty_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        return v.strip()


class QuestionResponse(BaseModel):
    """Schema for answer responses."""
    answer: str


class UploadResponse(BaseModel):
    """Schema for upload responses."""
    message: str
    filename: Optional[str] = None
    chunks_created: Optional[int] = None


class StatusResponse(BaseModel):
    """Schema for system and active report status."""
    status: str
    has_active_document: bool
    active_document_name: Optional[str] = None
    total_chunks: int = 0


class ErrorResponse(BaseModel):
    """Schema for structured error responses."""
    error: str
    detail: Optional[str] = None
