"""
PDF ingestion: text extraction and chunking.
 
Design considerations for chunking:
─────────────────────────────────────
1. **Recursive character splitting** — we split on paragraph boundaries first
   (double newlines), then sentences, then words. This preserves semantic
   coherence better than fixed-size windows.
 
2. **Overlap** — each chunk shares `CHUNK_OVERLAP` tokens with its neighbour
   so that context isn't lost at boundaries. A query whose answer spans two
   chunks will still find relevant content in at least one.
 
3. **Token estimation** — we approximate tokens as words × 1.3 (close to
   Mistral's BPE for English). This avoids pulling in a heavy tokenizer
   dependency while staying within budget.
 
4. **Metadata preservation** — every chunk knows its source file and page
   number, enabling precise citations in the response.
 
5. **Cleaning** — we strip excessive whitespace, headers/footers, and page
   numbers that add noise to retrieval.
"""
from __future__ import annotations
 
import logging
import re
import uuid
 
import fitz  # PyMuPDF
 
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_LENGTH
from app.models import Chunk
 
logger = logging.getLogger(__name__)
 
 
# ──────────────────────────────────────────────
# Text extraction
# ──────────────────────────────────────────────
 
def extract_text_from_pdf(pdf_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.
    Returns a list of {"page": int, "text": str}.
    """
    pages: list[dict] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            cleaned = _clean_text(text)
            if cleaned.strip():
                pages.append({"page": page_num + 1, "text": cleaned})
        doc.close()
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", filename, e)
        raise ValueError(f"Could not extract text from {filename}: {e}")
    return pages
 
 
def _clean_text(text: str) -> str:
    """Remove common PDF artefacts."""
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page-number-only lines
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Normalise whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)
 
 
# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
 
def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words × 1.3."""
    return int(len(text.split()) * 1.3)
 
 
def _split_recursive(
    text: str,
    max_tokens: int,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursively split text on progressively finer separators until each
    piece is under max_tokens.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]
 
    if _estimate_tokens(text) <= max_tokens:
        return [text]
 
    # Try the coarsest separator first
    for i, sep in enumerate(separators):
        parts = text.split(sep)
        if len(parts) == 1:
            continue  # this separator didn't split anything
 
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if _estimate_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If the single part is still too long, recurse with finer seps
                if _estimate_tokens(part) > max_tokens:
                    chunks.extend(
                        _split_recursive(part, max_tokens, separators[i + 1 :])
                    )
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks
 
    # Last resort: hard split by words
    words = text.split()
    chunks = []
    current_words: list[str] = []
    for w in words:
        current_words.append(w)
        if _estimate_tokens(" ".join(current_words)) >= max_tokens:
            chunks.append(" ".join(current_words))
            current_words = []
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks
 
 
def chunk_document(
    pages: list[dict],
    doc_id: str,
    filename: str,
) -> list[Chunk]:
    """
    Split extracted pages into overlapping chunks.
 
    Strategy:
    1. Concatenate page texts (keeping page boundaries noted).
    2. Recursively split into pieces ≤ CHUNK_SIZE tokens.
    3. Add overlap by prepending the tail of the previous chunk.
    """
    if not pages:
        return []
 
    # Build page-aware segments
    segments: list[tuple[str, int]] = []  # (text, page_number)
    for p in pages:
        pieces = _split_recursive(p["text"], CHUNK_SIZE)
        for piece in pieces:
            segments.append((piece, p["page"]))
 
    # Add overlap
    chunks: list[Chunk] = []
    prev_tail = ""
    for idx, (text, page_num) in enumerate(segments):
        if len(text.strip()) < MIN_CHUNK_LENGTH:
            continue
 
        # Prepend overlap from previous chunk
        if prev_tail and idx > 0:
            overlap_words = prev_tail.split()[-int(CHUNK_OVERLAP / 1.3) :]
            text_with_overlap = " ".join(overlap_words) + " " + text
        else:
            text_with_overlap = text
 
        chunk = Chunk(
            doc_id=doc_id,
            filename=filename,
            text=text_with_overlap.strip(),
            page_number=page_num,
            chunk_index=idx,
            token_count=_estimate_tokens(text_with_overlap),
        )
        chunks.append(chunk)
        prev_tail = text
 
    logger.info(
        "Chunked %s into %d chunks (from %d pages)",
        filename, len(chunks), len(pages),
    )
    return chunks