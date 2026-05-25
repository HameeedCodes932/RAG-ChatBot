"""
FastAPI application — entry point for the RAG system.
Serves the API endpoints and the frontend chatbot UI.
"""

import os
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingestion import ingestion_service
from rag_chain import rag_chain

# ── App setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Chatbot",
    description="RAG chatbot powered by LangChain & Groq (Llama 3.1 8B)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pre-warm on startup ───────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load embedding model and FAISS index so first query is fast."""
    print("⏳ Pre-loading embedding model...")
    _ = ingestion_service.embeddings   # loads all-MiniLM-L6-v2 into memory
    print("⏳ Pre-loading FAISS index...")
    _ = ingestion_service.vectorstore  # loads index from disk if exists
    print("✅ Ready — Groq (Llama 3.1 8B) + FAISS loaded.")

# ── Request / Response Models ─────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str

class UploadResponse(BaseModel):
    status: str
    message: str
    filename: str
    chunks: int
    characters: int

# ── API Endpoints ─────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Check system health and Groq connectivity."""
    llm_status = await rag_chain.health_check()
    vectorstore_ready = ingestion_service.vectorstore is not None
    docs = ingestion_service.get_ingested_documents()

    return {
        "status": "healthy",
        "llm": llm_status,
        "vectorstore_ready": vectorstore_ready,
        "documents_ingested": len(docs),
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a text, PDF, or Word file for ingestion into the RAG pipeline."""
    filename_lower = file.filename.lower()
    allowed_extensions = (".txt", ".pdf", ".docx")

    if not filename_lower.endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .pdf, and .docx files are supported.",
        )

    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {str(e)}")

    text = ""
    if filename_lower.endswith(".txt"):
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

    elif filename_lower.endswith(".pdf"):
        import io
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join(p.extract_text() for p in reader.pages if p.extract_text())
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")

    elif filename_lower.endswith(".docx"):
        import io
        try:
            import docx
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to parse Word document: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="The uploaded file has no extractable text content.")

    result = await ingestion_service.ingest_text(text, file.filename)

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    return UploadResponse(
        status=result["status"],
        message=result["message"],
        filename=result["filename"],
        chunks=result["chunks"],
        characters=result["characters"],
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a question to the RAG chatbot."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())
    result = await rag_chain.query(request.question, session_id)

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=result["session_id"],
    )


@app.get("/api/documents")
async def list_documents():
    """List all ingested documents."""
    docs = ingestion_service.get_ingested_documents()
    return {"documents": docs, "total": len(docs)}


@app.post("/api/clear-history")
async def clear_chat_history(session_id: str = "default"):
    """Clear chat history for a session."""
    rag_chain.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# ── Serve Frontend ───────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
