"""Unit tests for QuestionAnsweringService."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.services.qa_service import QuestionAnsweringService
from app.exceptions import LLMServiceError, DocumentNotUploadedError


def test_ask_without_document_uploaded(qa_service):
    """Asking a question before uploading document raises DocumentNotUploadedError."""
    with pytest.raises(DocumentNotUploadedError, match="No active financial report found"):
        qa_service.answer_question("What is the revenue?")


def test_build_prompt(qa_service, sample_documents):
    """Verifies prompt construction with context grounding instructions."""
    prompt = qa_service.build_prompt("What was Q4 revenue?", sample_documents)
    assert "You are a financial analyst assistant." in prompt
    assert "ACME Corp Q4 2023 Revenue" in prompt
    assert "What was Q4 revenue?" in prompt
    assert 'say "The report does not specify."' in prompt


def test_answer_question_success(qa_service, retrieval_service, sample_documents, mock_llm):
    """Verifies successful retrieval and LLM answer generation."""
    retrieval_service.create_index(sample_documents, "report.pdf")
    
    answer = qa_service.answer_question("What was the net income?")
    assert answer == "Total revenue for Q4 was $50 million with a 15% increase."
    assert len(mock_llm.invoked_prompts) == 1
    assert "ACME Corp" in mock_llm.invoked_prompts[0]


def test_answer_question_empty_retrieval(qa_service, retrieval_service, sample_documents):
    """Returns fallback when retriever returns empty list."""
    retrieval_service.create_index(sample_documents, "report.pdf")
    with patch.object(retrieval_service, "retrieve_context", return_value=[]):
        answer = qa_service.answer_question("Something completely absent?")
        assert answer == "The report does not specify."


def test_answer_question_llm_failure(qa_service, retrieval_service, sample_documents):
    """Translates raw LLM failures into LLMServiceError."""
    retrieval_service.create_index(sample_documents, "report.pdf")
    
    broken_llm = MagicMock()
    broken_llm.invoke.side_effect = RuntimeError("Groq Rate Limit Reached / API Key Invalid")
    qa_service._llm = broken_llm

    with pytest.raises(LLMServiceError, match="Failed to generate answer"):
        qa_service.answer_question("What is the guidance?")


def test_lazy_llm_initialization_missing_key(test_settings, retrieval_service):
    """Lazy LLM initialization raises error if GROQ_API_KEY is empty."""
    test_settings.groq_api_key = ""
    service = QuestionAnsweringService(retrieval_service=retrieval_service, llm=None, config=test_settings)
    
    with pytest.raises(LLMServiceError, match="GROQ_API_KEY.*is not configured"):
        _ = service.llm


def test_lazy_llm_initialization_with_key(test_settings, retrieval_service):
    """Lazy LLM initializes ChatGroq when API key is provided."""
    test_settings.groq_api_key = "gsk_test123"
    service = QuestionAnsweringService(retrieval_service=retrieval_service, llm=None, config=test_settings)
    with patch("app.services.qa_service.ChatGroq") as mock_chat:
        mock_chat.return_value = MagicMock()
        llm = service.llm
        assert llm is not None
        mock_chat.assert_called_once_with(
            groq_api_key="gsk_test123",
            model_name=test_settings.llm_model_name,
            temperature=test_settings.llm_temperature
        )
