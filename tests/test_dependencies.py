"""Unit tests for dependency injection getters."""
from app.dependencies import get_pdf_processor, get_retrieval_service, get_qa_service
from app.services.pdf_processor import PDFProcessor
from app.services.retrieval_service import RetrievalService
from app.services.qa_service import QuestionAnsweringService


def test_dependency_getters():
    """Verifies default dependency injection provider functions return valid service instances."""
    pdf_p = get_pdf_processor()
    assert isinstance(pdf_p, PDFProcessor)

    ret_s = get_retrieval_service()
    assert isinstance(ret_s, RetrievalService)

    qa_s = get_qa_service()
    assert isinstance(qa_s, QuestionAnsweringService)
