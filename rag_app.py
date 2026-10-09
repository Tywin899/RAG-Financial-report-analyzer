"""Standalone CLI interface for Financial Report Analyzer using refactored OOP services."""
import os
import sys
from dotenv import load_dotenv

from app.services.pdf_processor import PDFProcessor
from app.services.retrieval_service import RetrievalService
from app.services.qa_service import QuestionAnsweringService
from app.config import settings

load_dotenv()


def main():
    pdf_file = "financial_report.pdf"
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found.")
        sys.exit(1)

    print("Initializing services...")
    processor = PDFProcessor(config=settings)
    retrieval_service = RetrievalService(config=settings)
    qa_service = QuestionAnsweringService(retrieval_service=retrieval_service, config=settings)

    print(f"Loading and processing {pdf_file}...")
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()

    chunks = processor.process_pdf_bytes(pdf_bytes, filename=pdf_file)
    print(f"Extracted {len(chunks)} chunks.")

    print("Creating vector index...")
    retrieval_service.create_index(chunks, document_name=pdf_file)
    print("Vector index ready.")

    query = input("\nAsk a financial question: ")
    if not query.strip():
        print("Question cannot be empty.")
        return

    print("\nRetrieving context and generating answer...")
    answer = qa_service.answer_question(query)
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()