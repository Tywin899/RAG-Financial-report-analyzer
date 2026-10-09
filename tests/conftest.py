"""Test configuration, fixtures, and deterministic mocks."""
import io
import pytest
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.embeddings import Embeddings
from fastapi.testclient import TestClient

from app.config import Settings
from app.services.pdf_processor import PDFProcessor
from app.services.retrieval_service import RetrievalService
from app.services.qa_service import QuestionAnsweringService
from app.main import create_app
from app.dependencies import get_pdf_processor, get_retrieval_service, get_qa_service


class DeterministicMockEmbeddings(Embeddings):
    """Deterministic, lightweight offline embeddings for fast testing without HuggingFace."""
    def __init__(self, size: int = 384):
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Generate predictable normalized vectors based on text hash/length
        results = []
        for text in texts:
            val = float(len(text) % 10 + 1) / 10.0
            vec = [val] * self.size
            results.append(vec)
        return results

    def embed_query(self, text: str) -> List[float]:
        val = float(len(text) % 10 + 1) / 10.0
        return [val] * self.size


class MockChatLLM:
    """Mock LLM that returns predictable financial responses."""
    def __init__(self, return_text: str = "Total revenue for Q4 was $50 million with a 15% increase."):
        self.return_text = return_text
        self.invoked_prompts = []

    def invoke(self, prompt: str) -> AIMessage:
        self.invoked_prompts.append(prompt)
        return AIMessage(content=self.return_text)


@pytest.fixture
def test_settings() -> Settings:
    """Settings customized for testing."""
    return Settings(
        app_title="Test Financial API",
        chunk_size=100,
        chunk_overlap=20,
        retrieval_k=2,
        max_file_size_bytes=1024 * 1024,  # 1MB for test
        groq_api_key="test_mock_key"
    )


@pytest.fixture
def mock_embeddings() -> DeterministicMockEmbeddings:
    return DeterministicMockEmbeddings()


@pytest.fixture
def mock_llm() -> MockChatLLM:
    return MockChatLLM()


@pytest.fixture
def sample_documents() -> List[Document]:
    """Sample financial document chunks."""
    return [
        Document(page_content="ACME Corp Q4 2023 Revenue: $50 Million, Net Income: $10 Million.", metadata={"page": 1}),
        Document(page_content="Operating expenses increased by 5% year-over-year to $35 Million.", metadata={"page": 2}),
        Document(page_content="Guidance for 2024 projects 12% revenue growth driven by cloud division.", metadata={"page": 3}),
    ]


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Minimal valid PDF structure with extractable text."""
    # Constructing a minimal PDF with 1 page containing extractable text
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        b"4 0 obj << /Length 77 >> stream\n"
        b"BT /F1 12 Tf 72 712 Td (ACME Corp Q4 Revenue was $50 Million and Net Income was $10M.) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000373 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n448\n%%EOF\n"
    )
    return pdf_content


@pytest.fixture
def corrupted_pdf_bytes() -> bytes:
    return b"%PDF-corrupted-data-that-is-not-valid-pdf-structure"


@pytest.fixture
def pdf_processor(test_settings) -> PDFProcessor:
    return PDFProcessor(config=test_settings)


@pytest.fixture
def retrieval_service(mock_embeddings, test_settings) -> RetrievalService:
    return RetrievalService(embeddings=mock_embeddings, config=test_settings)


@pytest.fixture
def qa_service(retrieval_service, mock_llm, test_settings) -> QuestionAnsweringService:
    return QuestionAnsweringService(retrieval_service=retrieval_service, llm=mock_llm, config=test_settings)


@pytest.fixture
def test_app(pdf_processor, retrieval_service, qa_service):
    """FastAPI test app with mocked dependencies."""
    app = create_app()
    app.dependency_overrides[get_pdf_processor] = lambda: pdf_processor
    app.dependency_overrides[get_retrieval_service] = lambda: retrieval_service
    app.dependency_overrides[get_qa_service] = lambda: qa_service
    return app


@pytest.fixture
def client(test_app) -> TestClient:
    return TestClient(test_app, raise_server_exceptions=False)
