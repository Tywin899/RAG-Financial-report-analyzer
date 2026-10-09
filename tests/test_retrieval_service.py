"""Unit tests for RetrievalService."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.services.retrieval_service import RetrievalService
from app.exceptions import IndexingError, DocumentNotUploadedError


def test_initial_state_is_empty(retrieval_service):
    """Verifies that initially no vector index exists."""
    assert retrieval_service.has_index is False
    assert retrieval_service.active_document_name is None
    assert retrieval_service.chunk_count == 0


def test_create_index_success(retrieval_service, sample_documents):
    """Verifies successful index creation and state update."""
    count = retrieval_service.create_index(sample_documents, document_name="annual_report.pdf")
    assert count == 3
    assert retrieval_service.has_index is True
    assert retrieval_service.active_document_name == "annual_report.pdf"
    assert retrieval_service.chunk_count == 3


def test_create_index_empty_documents(retrieval_service):
    """Rejects creating index from empty document list."""
    with pytest.raises(IndexingError, match="empty document list"):
        retrieval_service.create_index([], document_name="empty.pdf")


def test_retrieve_context_success(retrieval_service, sample_documents):
    """Retrieves relevant document chunks for query."""
    retrieval_service.create_index(sample_documents, document_name="annual_report.pdf")
    results = retrieval_service.retrieve_context("What was the revenue?", k=2)
    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)


def test_retrieve_without_index_raises_error(retrieval_service):
    """Attempting retrieval before indexing raises DocumentNotUploadedError."""
    with pytest.raises(DocumentNotUploadedError, match="No active document index found"):
        retrieval_service.retrieve_context("Any question")


def test_index_replacement(retrieval_service, sample_documents):
    """Uploading a new document replaces previous index cleanly."""
    retrieval_service.create_index(sample_documents[:2], document_name="doc1.pdf")
    assert retrieval_service.chunk_count == 2
    assert retrieval_service.active_document_name == "doc1.pdf"

    # Replace with doc2
    retrieval_service.create_index(sample_documents, document_name="doc2.pdf")
    assert retrieval_service.chunk_count == 3
    assert retrieval_service.active_document_name == "doc2.pdf"


def test_state_preservation_on_indexing_failure(retrieval_service, sample_documents):
    """If second indexing fails, original valid index remains intact."""
    # First valid upload
    retrieval_service.create_index(sample_documents, document_name="valid.pdf")
    assert retrieval_service.has_index is True
    assert retrieval_service.active_document_name == "valid.pdf"

    # Second upload fails in FAISS.from_documents
    with patch("app.services.retrieval_service.FAISS.from_documents", side_effect=RuntimeError("FAISS CUDA out of memory")):
        with pytest.raises(IndexingError, match="Vector indexing failed"):
            retrieval_service.create_index([Document(page_content="Bad doc")], document_name="bad.pdf")

    # Verify original state was preserved
    assert retrieval_service.has_index is True
    assert retrieval_service.active_document_name == "valid.pdf"
    assert retrieval_service.chunk_count == 3


def test_clear_service(retrieval_service, sample_documents):
    """Clearing service resets index state."""
    retrieval_service.create_index(sample_documents, document_name="doc.pdf")
    assert retrieval_service.has_index is True

    retrieval_service.clear()
    assert retrieval_service.has_index is False
    assert retrieval_service.active_document_name is None
    assert retrieval_service.chunk_count == 0


def test_retrieve_context_exception_handling(retrieval_service, sample_documents):
    """Retrieval service translates retriever invocation errors to IndexingError."""
    retrieval_service.create_index(sample_documents, "doc.pdf")
    with patch.object(retrieval_service._vectorstore, "as_retriever", side_effect=RuntimeError("Index corrupt")):
        with pytest.raises(IndexingError, match="Failed to retrieve context"):
            retrieval_service.retrieve_context("query")


def test_lazy_embedding_initialization(test_settings):
    """Embeddings are lazily instantiated when not passed into constructor."""
    service = RetrievalService(embeddings=None, config=test_settings)
    with patch("app.services.retrieval_service.HuggingFaceEmbeddings") as mock_hf:
        mock_hf.return_value = MagicMock()
        emb = service.embeddings
        assert emb is not None
        mock_hf.assert_called_once_with(model_name=test_settings.embedding_model_name)
