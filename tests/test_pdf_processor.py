"""Unit tests for PDFProcessor."""
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.services.pdf_processor import PDFProcessor
from app.exceptions import (
    InvalidFileError,
    FileSizeExceededError,
    CorruptedPDFError,
    EmptyDocumentError
)


def test_validate_metadata_valid(pdf_processor):
    """Verifies that valid PDF metadata passes validation without error."""
    pdf_processor.validate_file_metadata("report.pdf", "application/pdf", size=5000)
    pdf_processor.validate_file_metadata("REPORT.PDF", "application/octet-stream", size=1000)


def test_validate_metadata_missing_filename(pdf_processor):
    """Rejects empty or missing filenames."""
    with pytest.raises(InvalidFileError, match="Filename is required"):
        pdf_processor.validate_file_metadata("", "application/pdf")


def test_validate_metadata_invalid_extension(pdf_processor):
    """Rejects non-PDF extensions like .exe or .txt."""
    with pytest.raises(InvalidFileError, match="Invalid file extension"):
        pdf_processor.validate_file_metadata("report.txt", "text/plain")


def test_validate_metadata_invalid_mime(pdf_processor):
    """Rejects invalid MIME type for PDF file."""
    with pytest.raises(InvalidFileError, match="Invalid MIME type"):
        pdf_processor.validate_file_metadata("report.pdf", "image/png")


def test_validate_metadata_file_too_large(pdf_processor):
    """Rejects file exceeding max file size limit."""
    with pytest.raises(FileSizeExceededError, match="exceeds the maximum limit"):
        pdf_processor.validate_file_metadata("huge_report.pdf", "application/pdf", size=2 * 1024 * 1024)


def test_process_pdf_bytes_empty_data(pdf_processor):
    """Rejects 0-byte upload."""
    with pytest.raises(EmptyDocumentError, match="empty"):
        pdf_processor.process_pdf_bytes(b"", "empty.pdf")


def test_process_pdf_bytes_invalid_header(pdf_processor):
    """Rejects files that do not start with %PDF- header."""
    with pytest.raises(InvalidFileError, match="valid PDF header"):
        pdf_processor.process_pdf_bytes(b"HELLO NOT A PDF", "fake.pdf")


def test_process_pdf_bytes_oversized(pdf_processor):
    """Rejects oversized bytes directly."""
    big_bytes = b"%PDF-" + b"0" * (2 * 1024 * 1024)
    with pytest.raises(FileSizeExceededError):
        pdf_processor.process_pdf_bytes(big_bytes, "oversized.pdf")


def test_process_valid_pdf_bytes(pdf_processor, sample_pdf_bytes):
    """Successfully processes valid PDF bytes and splits into chunks."""
    chunks = pdf_processor.process_pdf_bytes(sample_pdf_bytes, "valid_report.pdf")
    assert len(chunks) > 0
    assert any("ACME Corp" in chunk.page_content for chunk in chunks)


def test_process_corrupted_pdf_bytes(pdf_processor, corrupted_pdf_bytes):
    """Handles corrupted PDF bytes gracefully raising CorruptedPDFError."""
    with pytest.raises(CorruptedPDFError):
        pdf_processor.process_pdf_bytes(corrupted_pdf_bytes, "corrupt.pdf")


def test_process_pdf_empty_extracted_text(pdf_processor):
    """Handles PDFs with empty or whitespace-only pages."""
    with patch("app.services.pdf_processor.PyPDFLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        # Returns pages with only whitespace
        mock_loader.load.return_value = [Document(page_content="   \n\t  ")]
        mock_loader_cls.return_value = mock_loader

        with pytest.raises(EmptyDocumentError, match="no extractable text"):
            pdf_processor.process_pdf_path("dummy_path.pdf")


def test_process_pdf_no_pages(pdf_processor):
    """Handles PDFs with 0 pages loaded."""
    with patch("app.services.pdf_processor.PyPDFLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        with pytest.raises(EmptyDocumentError, match="No pages found"):
            pdf_processor.process_pdf_path("dummy_path.pdf")


def test_temp_file_cleanup_on_failure(pdf_processor):
    """Ensures temporary file is cleaned up even when PyPDFLoader throws exception."""
    created_temp_files = []

    original_named_temp_file = tempfile.NamedTemporaryFile

    def tracking_temp_file(*args, **kwargs):
        res = original_named_temp_file(*args, **kwargs)
        created_temp_files.append(res.name)
        return res

    with patch("tempfile.NamedTemporaryFile", side_effect=tracking_temp_file):
        with patch("app.services.pdf_processor.PyPDFLoader", side_effect=RuntimeError("Parsing crash")):
            with pytest.raises(CorruptedPDFError):
                pdf_processor.process_pdf_bytes(b"%PDF-1.4\nSome Content", "test.pdf")

    # Verify temp file was removed
    for temp_path in created_temp_files:
        assert not os.path.exists(temp_path), f"Temp file {temp_path} leaked!"


def test_process_pdf_empty_chunks_from_splitter(pdf_processor):
    """Handles edge case where text splitter produces no chunks."""
    with patch("app.services.pdf_processor.PyPDFLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="Valid Text")]
        mock_loader_cls.return_value = mock_loader

        with patch.object(pdf_processor.text_splitter, "split_documents", return_value=[]):
            with pytest.raises(EmptyDocumentError, match="Failed to generate document chunks"):
                pdf_processor.process_pdf_path("dummy.pdf")


def test_temp_file_cleanup_os_remove_failure(pdf_processor, sample_pdf_bytes):
    """Handles OS remove error without crashing the main flow."""
    with patch("os.remove", side_effect=OSError("Permission denied")):
        chunks = pdf_processor.process_pdf_bytes(sample_pdf_bytes, "report.pdf")
        assert len(chunks) > 0
