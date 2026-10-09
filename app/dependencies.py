"""Dependency injection providers for FastAPI."""
from app.config import settings
from app.services.pdf_processor import PDFProcessor
from app.services.retrieval_service import RetrievalService
from app.services.qa_service import QuestionAnsweringService

# Global service singletons (can be overridden in tests using app.dependency_overrides)
_pdf_processor = PDFProcessor(config=settings)
_retrieval_service = RetrievalService(config=settings)
_qa_service = QuestionAnsweringService(retrieval_service=_retrieval_service, config=settings)


def get_pdf_processor() -> PDFProcessor:
    """Provides PDFProcessor instance."""
    return _pdf_processor


def get_retrieval_service() -> RetrievalService:
    """Provides RetrievalService instance."""
    return _retrieval_service


def get_qa_service() -> QuestionAnsweringService:
    """Provides QuestionAnsweringService instance."""
    return _qa_service
