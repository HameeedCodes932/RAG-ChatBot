"""
RAG chain setup with LangChain + Groq.
Uses Groq's LPU inference for ultra-fast responses (200-500 tokens/sec).
"""

from collections import defaultdict
from typing import Optional

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    RAG_SYSTEM_PROMPT,
)
from ingestion import ingestion_service


class RAGChain:
    """Manages the RAG chain and per-session chat history."""

    def __init__(self):
        self._llm: Optional[ChatGroq] = None
        self._chat_history: dict[str, list] = defaultdict(list)

    @property
    def llm(self) -> ChatGroq:
        """Lazy-initialize the Groq client."""
        if self._llm is None:
            self._llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=0.1,       # lower = faster + more focused
                max_tokens=256,        # reduced from 1024 — enough for RAG answers
            )
        return self._llm

    def _build_prompt(self) -> ChatPromptTemplate:
        """Build the RAG prompt template with chat history support."""
        return ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

    def _build_chain(self):
        """Build and return a retrieval chain. Cached after first build."""
        retriever = ingestion_service.get_retriever()
        if retriever is None:
            return None
        prompt = self._build_prompt()
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(retriever, combine_docs_chain)

    async def query(self, question: str, session_id: str = "default") -> dict:
        """
        Run a RAG query: retrieve relevant chunks and generate an answer via Groq.
        """
        retriever = ingestion_service.get_retriever()

        if retriever is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload a text file first so I can answer questions about it.",
                "sources": [],
                "session_id": session_id,
            }

        # Build chain fresh each query (retriever may change after new uploads)
        chain = self._build_chain()

        # Get chat history for this session (keep last 3 exchanges = 6 messages)
        history = self._chat_history[session_id]

        try:
            result = await chain.ainvoke({
                "input": question,
                "chat_history": history,
            })
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return {
                    "answer": "⚠️ Invalid Groq API key. Please set your GROQ_API_KEY in config.py. Get a free key at console.groq.com",
                    "sources": [],
                    "session_id": session_id,
                }
            if "rate_limit" in error_msg.lower():
                return {
                    "answer": "⚠️ Groq rate limit hit. Please wait a moment and try again.",
                    "sources": [],
                    "session_id": session_id,
                }
            raise

        answer = result.get("answer", "I couldn't generate an answer.")

        # Extract source info
        sources = []
        for doc in result.get("context", []):
            source_info = {
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", -1),
            }
            sources.append(source_info)

        # Update chat history
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))

        # Keep only last 3 exchanges (6 messages) to keep prompts short
        if len(history) > 6:
            self._chat_history[session_id] = history[-6:]

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
        }

    def clear_history(self, session_id: str = "default"):
        """Clear chat history for a session."""
        self._chat_history[session_id] = []

    async def health_check(self) -> dict:
        """Check connectivity to Groq."""
        try:
            response = await self.llm.ainvoke("Say 'OK' in one word.")
            return {
                "status": "connected",
                "model": GROQ_MODEL,
                "provider": "Groq",
                "test_response": response.content[:100],
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "model": GROQ_MODEL,
                "provider": "Groq",
                "error": str(e),
            }


# Singleton instance
rag_chain = RAGChain()
