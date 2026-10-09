"""Retrieval and Vector Storage Service."""
import logging
import threading
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import Settings, settings as default_settings
from app.exceptions import IndexingError, DocumentNotUploadedError

logger = logging.getLogger(__name__)


class RetrievalService:
    """Manages document embeddings, FAISS indexing, similarity search, and atomic index lifecycle."""

    def __init__(
        self,
        embeddings: Optional[Any] = None,
        config: Optional[Settings] = None
    ):
        self.config = config or default_settings
        self._embeddings = embeddings
        self._vectorstore: Optional[FAISS] = None
        self._active_document_name: Optional[str] = None
        self._chunk_count: int = 0
        self._lock = threading.Lock()

    @property
    def embeddings(self) -> Any:
        """Lazy initializer for embedding model to avoid heavy startup overhead when not needed."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        return self._embeddings

    @property
    def has_index(self) -> bool:
        """Returns True if a vector index currently exists."""
        with self._lock:
            return self._vectorstore is not None

    @property
    def active_document_name(self) -> Optional[str]:
        """Returns the name of the currently indexed document."""
        with self._lock:
            return self._active_document_name

    @property
    def chunk_count(self) -> int:
        """Returns the number of chunks currently indexed."""
        with self._lock:
            return self._chunk_count

    def create_index(self, documents: List[Document], document_name: str = "document.pdf") -> int:
        """
        Creates a new FAISS vector index atomically.
        If creation fails, the previous index remains untouched.
        """
        if not documents:
            raise IndexingError("Cannot create vector index from empty document list.")

        try:
            # Build new index first without overwriting active state
            new_vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Atomic update under lock
            with self._lock:
                self._vectorstore = new_vectorstore
                self._active_document_name = document_name
                self._chunk_count = len(documents)
            
            logger.info(f"Successfully indexed {len(documents)} chunks for {document_name}")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise IndexingError(f"Vector indexing failed: {str(e)}")

    def retrieve_context(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieves top-k relevant document chunks for a query."""
        with self._lock:
            if self._vectorstore is None:
                raise DocumentNotUploadedError("No active document index found. Please upload a PDF first.")
            current_vectorstore = self._vectorstore

        top_k = k if k is not None else self.config.retrieval_k
        try:
            retriever = current_vectorstore.as_retriever(search_kwargs={"k": top_k})
            docs = retriever.invoke(query)
            return docs
        except Exception as e:
            logger.error(f"Error during context retrieval: {e}")
            raise IndexingError(f"Failed to retrieve context from index: {str(e)}")

    def clear(self) -> None:
        """Clears the active vector index and document metadata."""
        with self._lock:
            self._vectorstore = None
            self._active_document_name = None
            self._chunk_count = 0
            logger.info("Cleared active vector index.")
