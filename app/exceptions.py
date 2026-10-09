"""Domain-specific exceptions for Financial Report Analyzer."""


class FinancialReportAppException(Exception):
    """Base exception for the application."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InvalidFileError(FinancialReportAppException):
    """Raised when an uploaded file is not a valid PDF or is rejected."""
    def __init__(self, message: str = "Invalid file format. Only PDF files are supported."):
        super().__init__(message=message, status_code=400)


class FileSizeExceededError(FinancialReportAppException):
    """Raised when an uploaded file exceeds the allowed file size limit."""
    def __init__(self, message: str = "Uploaded file size exceeds the allowed limit."):
        super().__init__(message=message, status_code=413)


class CorruptedPDFError(FinancialReportAppException):
    """Raised when the uploaded PDF cannot be read or parsed."""
    def __init__(self, message: str = "Failed to parse PDF. The file may be corrupted or password protected."):
        super().__init__(message=message, status_code=422)


class EmptyDocumentError(FinancialReportAppException):
    """Raised when the PDF does not contain any extractable text."""
    def __init__(self, message: str = "The uploaded PDF contains no extractable text content."):
        super().__init__(message=message, status_code=422)


class DocumentNotUploadedError(FinancialReportAppException):
    """Raised when a user queries before uploading a document."""
    def __init__(self, message: str = "Please upload a financial report PDF first."):
        super().__init__(message=message, status_code=400)


class IndexingError(FinancialReportAppException):
    """Raised when vector store indexing fails."""
    def __init__(self, message: str = "Failed to index document in vector store."):
        super().__init__(message=message, status_code=500)


class LLMServiceError(FinancialReportAppException):
    """Raised when the language model invocation fails."""
    def __init__(self, message: str = "LLM generation service encountered an error."):
        super().__init__(message=message, status_code=502)
