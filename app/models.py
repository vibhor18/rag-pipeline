"""
Pydantic models for request/response schemas and internal data structures.
"""
from __future__ import annotations
 
import uuid
from dataclasses import dataclass, field
from typing import Literal
 
from pydantic import BaseModel
 
 
# ──────────────────────────────────────────────
# Internal data structures (not exposed via API)
# ──────────────────────────────────────────────
 
@dataclass
class Chunk:
    """A single chunk of text extracted from a document."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    doc_id: str = ""
    filename: str = ""
    text: str = ""
    page_number: int = 0
    chunk_index: int = 0
    embedding: list[float] = field(default_factory=list)
    token_count: int = 0
 
 
@dataclass
class SearchResult:
    """A scored chunk returned from search."""
    chunk: Chunk
    score: float = 0.0
    method: str = ""  # "semantic", "bm25", or "hybrid"
 
 
# ──────────────────────────────────────────────
# API request / response models
# ──────────────────────────────────────────────
 
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    conversation_id: str | None = None
 
 
class Citation(BaseModel):
    chunk_id: str
    filename: str
    page_number: int
    text_snippet: str
    relevance_score: float
 
 
class QueryResponse(BaseModel):
    answer: str
    intent: str
    citations: list[Citation] = []
    query_transformed: str = ""
    warning: str | None = None
 
 
class IngestionResponse(BaseModel):
    status: str
    documents_ingested: int
    total_chunks: int
    details: list[dict]
 
 
class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int
    num_pages: int
 
 
class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    total_chunks: int
 
 
class IntentResult(BaseModel):
    intent: Literal["greeting", "rag", "refusal", "chitchat"]
    reason: str
    pii_detected: bool = False