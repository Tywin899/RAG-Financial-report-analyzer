"""FastAPI application factory and route definitions."""
import logging
from fastapi import FastAPI, UploadFile, File, Depends, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.models import QuestionRequest, QuestionResponse, UploadResponse, StatusResponse, ErrorResponse
from app.exceptions import FinancialReportAppException, DocumentNotUploadedError
from app.dependencies import (
    get_pdf_processor,
    get_retrieval_service,
    get_qa_service,
    PDFProcessor,
    RetrievalService,
    QuestionAnsweringService
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application instance."""
    app = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        description="Object-Oriented RAG-based Financial Report Analyzer API"
    )

    # Global Exception Handlers
    @app.exception_handler(FinancialReportAppException)
    async def app_exception_handler(request: Request, exc: FinancialReportAppException):
        logger.warning(f"Application exception on {request.url.path}: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        msg = exc.errors()[0].get("msg", "Request validation failed") if exc.errors() else "Request validation failed"
        return JSONResponse(
            status_code=422,
            content={"error": msg, "detail": str(exc.errors())}
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An unexpected server error occurred."}
        )

    # Routes
    @app.get("/health", status_code=status.HTTP_200_OK)
    async def health_check():
        """Service health check endpoint."""
        return {"status": "healthy", "version": settings.app_version}

    @app.get("/status", response_model=StatusResponse, status_code=status.HTTP_200_OK)
    async def get_status(
        retrieval_service: RetrievalService = Depends(get_retrieval_service)
    ):
        """Returns the active report status and chunk counts."""
        return StatusResponse(
            status="active" if retrieval_service.has_index else "idle",
            has_active_document=retrieval_service.has_index,
            active_document_name=retrieval_service.active_document_name,
            total_chunks=retrieval_service.chunk_count
        )

    @app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_200_OK)
    async def upload_pdf(
        file: UploadFile = File(...),
        pdf_processor: PDFProcessor = Depends(get_pdf_processor),
        retrieval_service: RetrievalService = Depends(get_retrieval_service)
    ):
        """
        Validates, extracts, splits, and indexes an uploaded financial report PDF.
        Replaces previous report upon success, or preserves previous report on failure.
        """
        # Validate metadata
        pdf_processor.validate_file_metadata(
            filename=file.filename,
            content_type=file.content_type,
            size=file.size
        )

        # Read contents
        file_bytes = await file.read()
        
        # Process and chunk PDF
        chunks = pdf_processor.process_pdf_bytes(file_bytes, filename=file.filename or "document.pdf")
        
        # Create atomic index
        chunks_count = retrieval_service.create_index(chunks, document_name=file.filename or "document.pdf")

        return UploadResponse(
            message="PDF processed successfully",
            filename=file.filename,
            chunks_created=chunks_count
        )

    @app.post("/ask", response_model=QuestionResponse, status_code=status.HTTP_200_OK)
    async def ask_question(
        request: QuestionRequest,
        qa_service: QuestionAnsweringService = Depends(get_qa_service)
    ):
        """
        Retrieves grounded financial context and generates an answer using Groq LLM.
        """
        answer = qa_service.answer_question(request.question)
        return QuestionResponse(answer=answer)

    @app.post("/reset", status_code=status.HTTP_200_OK)
    async def reset_session(
        retrieval_service: RetrievalService = Depends(get_retrieval_service)
    ):
        """Resets the vector index and clears active document state."""
        retrieval_service.clear()
        return {"message": "Session reset successfully"}

    return app


app = create_app()
