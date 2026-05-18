"""
RAG Pipeline — FastAPI Application
 
Endpoints:
  POST /api/ingest       Upload and ingest PDF files
  POST /api/query        Query the knowledge base
  GET  /api/documents    List ingested documents
  DELETE /api/documents/{doc_id}  Remove a document
  GET  /api/health       Health check
  GET  /                 Chat UI
"""
from __future__ import annotations
 
import logging
import uuid
from pathlib import Path
 
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
 
from app.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_FILES_PER_UPLOAD
from app.generation import generate_response
from app.ingestion import chunk_document, extract_text_from_pdf
from app.mistral_client import close_client, embed_texts, embed_query
from app.models import (
    HealthResponse,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)
from app.query_processing import detect_intent, transform_query
from app.vector_store import store
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
 
app = FastAPI(
    title="RAG Pipeline",
    description="A Retrieval-Augmented Generation system for PDF documents",
    version="1.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Serve static files
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
 
 
# ──────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────
 
@app.on_event("shutdown")
async def shutdown():
    await close_client()
 
 
# ──────────────────────────────────────────────
# Ingestion endpoint
# ──────────────────────────────────────────────
 
@app.post("/api/ingest", response_model=IngestionResponse)
async def ingest_files(files: list[UploadFile] = File(...)):
    """
    Upload one or more PDF files for ingestion.
    Extracts text, chunks it, embeds it, and stores in the vector store.
    """
    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_FILES_PER_UPLOAD} files per upload.",
        )
 
    details = []
    total_chunks = 0
 
    for file in files:
        # Validate extension
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            details.append(
                {"filename": file.filename, "status": "skipped", "reason": "Not a PDF"}
            )
            continue
 
        # Read file
        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            details.append(
                {
                    "filename": file.filename,
                    "status": "skipped",
                    "reason": f"File too large ({size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB)",
                }
            )
            continue
 
        try:
            # Extract text
            doc_id = uuid.uuid4().hex[:16]
            pages = extract_text_from_pdf(content, file.filename or "unknown.pdf")
 
            if not pages:
                details.append(
                    {"filename": file.filename, "status": "skipped", "reason": "No text extracted"}
                )
                continue
 
            # Chunk
            chunks = chunk_document(pages, doc_id, file.filename or "unknown.pdf")
 
            if not chunks:
                details.append(
                    {"filename": file.filename, "status": "skipped", "reason": "No valid chunks"}
                )
                continue
 
            # Embed
            texts_to_embed = [c.text for c in chunks]
            embeddings = await embed_texts(texts_to_embed)
 
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
 
            # Store
            store.add_chunks(chunks)
            total_chunks += len(chunks)
 
            details.append(
                {
                    "filename": file.filename,
                    "status": "success",
                    "doc_id": doc_id,
                    "pages": len(pages),
                    "chunks": len(chunks),
                }
            )
            logger.info(
                "Ingested %s: %d pages, %d chunks",
                file.filename, len(pages), len(chunks),
            )
 
        except Exception as e:
            logger.error("Failed to ingest %s: %s", file.filename, e)
            details.append(
                {"filename": file.filename, "status": "error", "reason": str(e)}
            )
 
    return IngestionResponse(
        status="completed",
        documents_ingested=sum(1 for d in details if d["status"] == "success"),
        total_chunks=total_chunks,
        details=details,
    )
 
 
# ──────────────────────────────────────────────
# Query endpoint
# ──────────────────────────────────────────────
 
@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the knowledge base with a user question.
    Pipeline: intent detection → query transform → hybrid search → generation.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
 
    # Step 1: Intent detection
    intent_result = await detect_intent(question)
    logger.info("Intent: %s (%s)", intent_result.intent, intent_result.reason)
 
    # Short-circuit for non-RAG intents
    if intent_result.intent != "rag":
        return await generate_response(
            question=question,
            intent=intent_result.intent,
            search_results=[],
            has_documents=store.num_documents > 0,
        )
 
    # Check if we have documents
    if store.num_documents == 0:
        return await generate_response(
            question=question,
            intent="rag",
            search_results=[],
            has_documents=False,
        )
 
    # Step 2: Query transformation
    transformed = await transform_query(question)
    logger.info("Transformed query: %s", transformed)
 
    # Step 3: Embed the transformed query
    query_emb = await embed_query(transformed)
 
    # Step 4: Hybrid search
    results = store.hybrid_search(
        query=transformed,
        query_embedding=query_emb,
        top_k=request.top_k,
    )
    logger.info("Retrieved %d results", len(results))
 
    # Step 5: Generate response
    response = await generate_response(
        question=question,
        intent="rag",
        search_results=results,
        transformed_query=transformed,
        has_documents=True,
    )
    return response
 
 
# ──────────────────────────────────────────────
# Document management
# ──────────────────────────────────────────────
 
@app.get("/api/documents")
async def list_documents():
    """List all ingested documents."""
    return {"documents": store.get_documents()}
 
 
@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document and its chunks."""
    if store.delete_document(doc_id):
        return {"status": "deleted", "doc_id": doc_id}
    raise HTTPException(status_code=404, detail="Document not found.")
 
 
@app.delete("/api/documents")
async def clear_all():
    """Delete all documents."""
    store.clear()
    return {"status": "cleared"}
 
 
# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
 
@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        documents_loaded=store.num_documents,
        total_chunks=store.num_chunks,
    )
 
 
# ──────────────────────────────────────────────
# Serve the chat UI
# ──────────────────────────────────────────────
 
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return HTMLResponse(html_path.read_text())
 