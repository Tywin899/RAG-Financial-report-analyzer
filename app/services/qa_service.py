"""Question Answering and Prompt Orchestration Service."""
import logging
from typing import Optional, Any, List
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from app.config import Settings, settings as default_settings
from app.exceptions import LLMServiceError, DocumentNotUploadedError
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class QuestionAnsweringService:
    """Orchestrates context retrieval, prompt construction, and grounded LLM generation."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm: Optional[Any] = None,
        config: Optional[Settings] = None
    ):
        self.retrieval_service = retrieval_service
        self.config = config or default_settings
        self._llm = llm

    @property
    def llm(self) -> Any:
        """Lazy initializer for Groq Chat model."""
        if self._llm is None:
            if not self.config.groq_api_key:
                raise LLMServiceError("GROQ_API_KEY environment variable is not configured.")
            self._llm = ChatGroq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.llm_model_name,
                temperature=self.config.llm_temperature
            )
        return self._llm

    def build_prompt(self, question: str, context_docs: List[Document]) -> str:
        """Builds the prompt instructing the LLM to answer strictly from retrieved context."""
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""You are a financial analyst assistant.

Answer the question using ONLY the provided financial report context below.
If the answer is not present in the context, explicitly say "The report does not specify."

Financial Report Context:
{context}

Question:
{question}
"""
        return prompt

    def answer_question(self, question: str) -> str:
        """
        Retrieves context, formats prompt, and invokes LLM.
        Raises DocumentNotUploadedError if no document is uploaded,
        or LLMServiceError if LLM invocation fails.
        """
        if not self.retrieval_service.has_index:
            raise DocumentNotUploadedError("No active financial report found. Please upload a PDF first.")

        relevant_docs = self.retrieval_service.retrieve_context(question)
        
        if not relevant_docs:
            return "The report does not specify."

        prompt = self.build_prompt(question, relevant_docs)

        try:
            response = self.llm.invoke(prompt)
            # Response can be AIMessage or string or mock
            content = getattr(response, "content", str(response))
            return content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMServiceError(f"Failed to generate answer from language model: {str(e)}")
