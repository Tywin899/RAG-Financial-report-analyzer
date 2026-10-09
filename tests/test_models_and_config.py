"""Unit tests for Pydantic Models and Configuration."""
import pytest
from pydantic import ValidationError

from app.models import QuestionRequest, QuestionResponse, UploadResponse, StatusResponse, ErrorResponse
from app.config import Settings
from app.exceptions import (
    FinancialReportAppException,
    InvalidFileError,
    FileSizeExceededError,
    CorruptedPDFError,
    EmptyDocumentError,
    DocumentNotUploadedError,
    IndexingError,
    LLMServiceError,
)


def test_question_request_valid():
    req = QuestionRequest(question="  What was the Q3 gross margin?  ")
    assert req.question == "What was the Q3 gross margin?"


def test_question_request_empty_or_whitespace():
    with pytest.raises(ValidationError):
        QuestionRequest(question="")

    with pytest.raises(ValidationError):
        QuestionRequest(question="    \n\t  ")


def test_models_serialization():
    q_resp = QuestionResponse(answer="Profit rose 10%")
    assert q_resp.model_dump() == {"answer": "Profit rose 10%"}

    up_resp = UploadResponse(message="OK", filename="f.pdf", chunks_created=5)
    assert up_resp.chunks_created == 5

    st_resp = StatusResponse(status="active", has_active_document=True, active_document_name="f.pdf", total_chunks=5)
    assert st_resp.has_active_document is True

    err_resp = ErrorResponse(error="Err", detail="detail")
    assert err_resp.error == "Err"


def test_custom_exceptions_properties():
    ex = InvalidFileError("custom message")
    assert ex.status_code == 400
    assert ex.message == "custom message"

    ex2 = FileSizeExceededError()
    assert ex2.status_code == 413

    ex3 = CorruptedPDFError()
    assert ex3.status_code == 422

    ex4 = EmptyDocumentError()
    assert ex4.status_code == 422

    ex5 = DocumentNotUploadedError()
    assert ex5.status_code == 400

    ex6 = IndexingError()
    assert ex6.status_code == 500

    ex7 = LLMServiceError()
    assert ex7.status_code == 502


def test_config_settings_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL_NAME", raising=False)
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    cfg = Settings()
    assert cfg.chunk_size == 1000
    assert cfg.chunk_overlap == 200
    assert cfg.retrieval_k == 4
    assert cfg.llm_model_name == "openai/gpt-oss-20b"
    assert ".pdf" in cfg.allowed_extensions
    assert "application/pdf" in cfg.allowed_mime_types


def test_config_settings_env_override(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL_NAME", "custom-groq-model")
    cfg = Settings()
    assert cfg.llm_model_name == "custom-groq-model"

