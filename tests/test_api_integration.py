"""Integration tests for FastAPI endpoints using TestClient."""
import io
import pytest
from unittest.mock import patch
from langchain_core.documents import Document

from app.exceptions import IndexingError, LLMServiceError


def test_health_endpoint(client):
    """GET /health returns healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_status_endpoint_empty(client):
    """GET /status initially indicates idle with no active document."""
    res = client.get("/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "idle"
    assert data["has_active_document"] is False
    assert data["total_chunks"] == 0


def test_ask_before_upload(client):
    """POST /ask before uploading returns 400 Bad Request."""
    res = client.post("/ask", json={"question": "What is the net income?"})
    assert res.status_code == 400
    assert "upload a PDF first" in res.json()["error"]


def test_ask_empty_question(client):
    """POST /ask with empty or whitespace question returns 422 Unprocessable Entity."""
    res = client.post("/ask", json={"question": "   "})
    assert res.status_code == 422
    assert "error" in res.json()


def test_ask_missing_field(client):
    """POST /ask with missing question field returns 422."""
    res = client.post("/ask", json={})
    assert res.status_code == 422


def test_upload_invalid_file_extension(client):
    """POST /upload with non-PDF extension is rejected with 400."""
    file_content = b"Some plain text file"
    res = client.post(
        "/upload",
        files={"file": ("report.txt", file_content, "text/plain")}
    )
    assert res.status_code == 400
    assert "Invalid file extension" in res.json()["error"]


def test_upload_corrupted_pdf(client, corrupted_pdf_bytes):
    """POST /upload with corrupted PDF structure returns 422."""
    res = client.post(
        "/upload",
        files={"file": ("corrupt.pdf", corrupted_pdf_bytes, "application/pdf")}
    )
    assert res.status_code == 422
    assert "Failed to parse PDF" in res.json()["error"] or "error" in res.json()


def test_upload_and_ask_flow(client, sample_pdf_bytes):
    """Verifies end-to-end flow: Upload PDF -> Check Status -> Ask Question -> Reset Session."""
    # 1. Upload valid PDF
    upload_res = client.post(
        "/upload",
        files={"file": ("q4_report.pdf", sample_pdf_bytes, "application/pdf")}
    )
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["message"] == "PDF processed successfully"
    assert upload_data["chunks_created"] > 0
    assert upload_data["filename"] == "q4_report.pdf"

    # 2. Check status shows active report
    status_res = client.get("/status")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["has_active_document"] is True
    assert status_data["active_document_name"] == "q4_report.pdf"
    assert status_data["total_chunks"] == upload_data["chunks_created"]

    # 3. Ask question
    ask_res = client.post("/ask", json={"question": "What is the revenue?"})
    assert ask_res.status_code == 200
    ask_data = ask_res.json()
    assert "answer" in ask_data
    assert len(ask_data["answer"]) > 0

    # 4. Reset session
    reset_res = client.post("/reset")
    assert reset_res.status_code == 200
    assert client.get("/status").json()["has_active_document"] is False


def test_upload_second_report_replaces_first(client, sample_pdf_bytes):
    """Uploading a second valid report replaces the active document."""
    # Upload first
    client.post("/upload", files={"file": ("doc1.pdf", sample_pdf_bytes, "application/pdf")})
    assert client.get("/status").json()["active_document_name"] == "doc1.pdf"

    # Upload second
    client.post("/upload", files={"file": ("doc2.pdf", sample_pdf_bytes, "application/pdf")})
    assert client.get("/status").json()["active_document_name"] == "doc2.pdf"


def test_failed_second_upload_preserves_first_report(client, sample_pdf_bytes, corrupted_pdf_bytes):
    """If a second upload fails, the first report remains active and intact."""
    # Upload first valid
    client.post("/upload", files={"file": ("first_valid.pdf", sample_pdf_bytes, "application/pdf")})
    assert client.get("/status").json()["active_document_name"] == "first_valid.pdf"

    # Try uploading corrupt second
    bad_res = client.post("/upload", files={"file": ("bad.pdf", corrupted_pdf_bytes, "application/pdf")})
    assert bad_res.status_code == 422

    # Verify first report is still active
    status = client.get("/status").json()
    assert status["has_active_document"] is True
    assert status["active_document_name"] == "first_valid.pdf"


def test_llm_failure_handling(client, sample_pdf_bytes, qa_service):
    """LLM failure returns 502 with structured error."""
    client.post("/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")})
    
    with patch.object(qa_service, "answer_question", side_effect=LLMServiceError("Groq service unavailable")):
        res = client.post("/ask", json={"question": "What is the guidance?"})
        assert res.status_code == 502
        assert "Groq service unavailable" in res.json()["error"]


def test_six_step_state_and_failure_recovery_flow(client, sample_pdf_bytes, corrupted_pdf_bytes):
    """
    Explicitly tests the 6-step lifecycle requested:
    1. Upload a valid PDF.
    2. Confirm that the document index is available.
    3. Attempt to upload an invalid or corrupted PDF.
    4. Confirm that the previous valid index remains available.
    5. Ask a question and confirm that the previous document can still be used.
    6. Upload another valid PDF and confirm that the active index is replaced as intended.
    """
    # Step 1: Upload a valid PDF
    res1 = client.post("/upload", files={"file": ("report_q1.pdf", sample_pdf_bytes, "application/pdf")})
    assert res1.status_code == 200

    # Step 2: Confirm document index is available
    status1 = client.get("/status").json()
    assert status1["has_active_document"] is True
    assert status1["active_document_name"] == "report_q1.pdf"
    assert status1["total_chunks"] > 0

    # Step 3: Attempt to upload an invalid/corrupted PDF
    res2 = client.post("/upload", files={"file": ("corrupt_q2.pdf", corrupted_pdf_bytes, "application/pdf")})
    assert res2.status_code == 422

    # Step 4: Confirm previous valid index remains available
    status2 = client.get("/status").json()
    assert status2["has_active_document"] is True
    assert status2["active_document_name"] == "report_q1.pdf"
    assert status2["total_chunks"] == status1["total_chunks"]

    # Step 5: Ask a question and confirm previous document can still be used
    ask_res = client.post("/ask", json={"question": "What is the net revenue?"})
    assert ask_res.status_code == 200
    assert "answer" in ask_res.json()

    # Step 6: Upload another valid PDF and confirm active index is replaced
    res3 = client.post("/upload", files={"file": ("report_q2_new.pdf", sample_pdf_bytes, "application/pdf")})
    assert res3.status_code == 200
    status3 = client.get("/status").json()
    assert status3["has_active_document"] is True
    assert status3["active_document_name"] == "report_q2_new.pdf"


def test_generic_unhandled_exception(client, sample_pdf_bytes, qa_service):
    """Unhandled generic exception returns 500 without leaking stack traces."""
    client.post("/upload", files={"file": ("report.pdf", sample_pdf_bytes, "application/pdf")})
    
    with patch.object(qa_service, "answer_question", side_effect=RuntimeError("Secret database connection string failure")):
        res = client.post("/ask", json={"question": "Crash the server"})
        assert res.status_code == 500
        assert "An unexpected server error occurred." in res.json()["error"]
        assert "Secret" not in res.json()["error"]
