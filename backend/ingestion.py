"""
Document ingestion pipeline.
Handles text file processing, chunking, embedding, and FAISS storage.
"""

import os
import time
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
)


class DocumentIngestion:
    """Manages document ingestion into a FAISS vector store."""

    def __init__(self):
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[FAISS] = None
        self._ingested_docs: list[dict] = []

        # Text splitter configuration
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load the embedding model (first call downloads ~80MB)."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    @property
    def vectorstore(self) -> Optional[FAISS]:
        """Return the current FAISS vectorstore, loading from disk if available."""
        if self._vectorstore is None:
            self._load_from_disk()
        return self._vectorstore

    def _load_from_disk(self):
        """Attempt to load a persisted FAISS index from disk."""
        index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
        if os.path.exists(index_path):
            try:
                self._vectorstore = FAISS.load_local(
                    FAISS_INDEX_DIR,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                print(f"[WARNING] Could not load FAISS index from disk: {e}")

    def _save_to_disk(self):
        """Persist the FAISS index to disk."""
        if self._vectorstore is not None:
            os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
            self._vectorstore.save_local(FAISS_INDEX_DIR)

    async def ingest_text(self, text: str, filename: str) -> dict:
        """
        Ingest a text string: split into chunks, embed, and store in FAISS.

        Args:
            text: The raw text content from the uploaded file.
            filename: Original filename for metadata tracking.

        Returns:
            dict with ingestion results (chunk count, filename, etc.)
        """
        # Split text into chunks
        chunks = self._splitter.split_text(text)

        if not chunks:
            return {
                "status": "error",
                "message": "No text content found in the file.",
                "filename": filename,
                "chunks": 0,
            }

        # Add metadata to each chunk
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        # Create or merge into vectorstore
        if self._vectorstore is None:
            self._vectorstore = FAISS.from_texts(
                chunks, self.embeddings, metadatas=metadatas
            )
        else:
            new_store = FAISS.from_texts(
                chunks, self.embeddings, metadatas=metadatas
            )
            self._vectorstore.merge_from(new_store)

        # Persist to disk
        self._save_to_disk()

        # Track ingested document
        doc_info = {
            "filename": filename,
            "chunks": len(chunks),
            "characters": len(text),
            "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._ingested_docs.append(doc_info)

        return {
            "status": "success",
            "message": f"Successfully ingested '{filename}'",
            **doc_info,
        }

    def get_ingested_documents(self) -> list[dict]:
        """Return the list of ingested documents."""
        return self._ingested_docs

    def get_retriever(self, k: int = 4):
        """
        Return a FAISS retriever for the RAG chain.

        Args:
            k: Number of top chunks to retrieve.
        """
        if self._vectorstore is None:
            return None
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )


# Singleton instance
ingestion_service = DocumentIngestion()
