"""
Configuration for the RAG system.
Centralizes all settings for Groq, embeddings, and vector store.
"""

import os

# ── Groq API ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Embedding Model ──────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── FAISS Vector Store ───────────────────────────────────────────────
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

# ── Text Splitting ───────────────────────────────────────────────────
CHUNK_SIZE    = 500   # reduced from 1000 for faster context processing
CHUNK_OVERLAP = 50    # reduced from 150

# ── RAG Prompt ───────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
Use ONLY the following context to answer the user's question.
If the answer cannot be found in the context, say "I don't have enough information in the uploaded documents to answer that question."
Be concise and accurate.

Context:
{context}"""
