"""PDF Processing and validation service."""
import os
import tempfile
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, settings as default_settings
from app.exceptions import (
    InvalidFileError,
    FileSizeExceededError,
    CorruptedPDFError,
    EmptyDocumentError,
    FinancialReportAppException
)

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF validation, secure temporary storage, text extraction, and chunking."""

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or default_settings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def validate_file_metadata(self, filename: Optional[str], content_type: Optional[str], size: Optional[int] = None) -> None:
        """Validates filename extension, MIME type, and size."""
        if not filename:
            raise InvalidFileError("Filename is required.")

        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.config.allowed_extensions:
            raise InvalidFileError(f"Invalid file extension '{ext}'. Only {self.config.allowed_extensions} are accepted.")

        if content_type and content_type.lower() not in self.config.allowed_mime_types and content_type != "application/octet-stream":
            raise InvalidFileError(f"Invalid MIME type '{content_type}'. Must be application/pdf.")

        if size is not None and size > self.config.max_file_size_bytes:
            raise FileSizeExceededError(
                f"File size {size} bytes exceeds the maximum limit of {self.config.max_file_size_bytes} bytes."
            )

    def process_pdf_bytes(self, file_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
        """Processes raw bytes of a PDF and returns split LangChain Document chunks."""
        if not file_bytes or len(file_bytes) == 0:
            raise EmptyDocumentError("The uploaded file is empty (0 bytes).")

        if len(file_bytes) > self.config.max_file_size_bytes:
            raise FileSizeExceededError("File size exceeds maximum allowed limit.")

        # Check basic PDF magic header
        if not file_bytes.startswith(b"%PDF-"):
            raise InvalidFileError("File does not have a valid PDF header.")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            return self.process_pdf_path(tmp_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {tmp_path}: {e}")

    def process_pdf_path(self, pdf_path: str) -> List[Document]:
        """Extracts text from a PDF on disk and chunks it."""
        try:
            loader = PyPDFLoader(pdf_path)
            raw_docs = loader.load()
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise CorruptedPDFError(f"Unable to read or parse PDF file: {str(e)}")

        if not raw_docs:
            raise EmptyDocumentError("No pages found in the uploaded PDF.")

        # Ensure that at least one page has extractable non-whitespace text
        total_text_length = sum(len(doc.page_content.strip()) for doc in raw_docs)
        if total_text_length == 0:
            raise EmptyDocumentError("The uploaded PDF contains no extractable text.")

        chunks = self.text_splitter.split_documents(raw_docs)
        if not chunks:
            raise EmptyDocumentError("Failed to generate document chunks from extracted text.")

        return chunks
