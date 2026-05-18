# RAG Pipeline — Document Q&A

A Retrieval-Augmented Generation system that lets users upload PDF documents and ask questions about their content. Built from scratch with FastAPI and Mistral AI — no external RAG libraries, no third-party vector databases. All search and retrieval logic is implemented manually using numpy.

## Architecture

The system follows a 5-stage pipeline:
### 1. Data Ingestion
- PDF text extraction using PyMuPDF (page-by-page with metadata preservation)
- Recursive character splitting — splits on paragraph boundaries first, then sentences, then words. Preserves semantic coherence better than fixed-size windows
- 512-token chunks with 64-token overlap so context isn't lost at boundaries
- Text cleaning (strips excessive whitespace, page numbers, headers/footers)
- Batch embedding via Mistral's `mistral-embed` model (1024 dimensions)

### 2. Query Processing
- **PII detection** — regex patterns catch SSNs, credit card numbers, and passport-like strings before they reach the LLM
- **Intent classification** — Mistral classifies each query as `greeting`, `chitchat`, `rag`, or `refusal`. Non-RAG intents are short-circuited without searching
- **Query transformation** — rewrites the user's question for better retrieval: expands abbreviations, adds synonyms, removes filler words

### 3. Hybrid Search
Two search methods run in parallel, then get merged:
- **Semantic search** — cosine similarity between the query embedding and all stored chunk embeddings (numpy dot product on L2-normalised vectors)
- **BM25 keyword search** — implemented from scratch with TF-IDF scoring, IDF smoothing, document length normalisation, and stopword removal

Results are combined using **Reciprocal Rank Fusion (RRF)** rather than linear score interpolation. RRF uses rank positions instead of raw scores, which avoids the problem of BM25 and cosine similarity being on completely different scales.

### 4. Re-ranking & Filtering
- RRF merges and re-ranks results from both search methods
- A similarity threshold (0.35 cosine minimum) filters out low-confidence chunks
- If no chunks pass the threshold, the system returns "insufficient evidence" instead of hallucinating

### 5. Generation
- **Answer shaping** — detects question type (list, comparison, general) and selects the appropriate prompt template
- Mistral LLM generates the answer grounded in the retrieved context
- **Post-hoc hallucination filter** — a second LLM call checks each sentence of the answer against the evidence, marking them as SUPPORTED, UNSUPPORTED, or NEUTRAL. Unsupported sentences are stripped from the final response
- Citations are built from the retrieved chunks with filename, page number, and relevance score

## Setup

```bash
git clone https://github.com/vibhor18/rag-pipeline.git
cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add Mistral API key from https://console.mistral.ai/

# Run the server
uvicorn app.main:app --reload

# Open http://localhost:8000 in your browser
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/ingest` | Upload one or more PDF files for ingestion |
| `POST` | `/api/query` | Ask a question about uploaded documents |
| `GET` | `/api/documents` | List all ingested documents |
| `DELETE` | `/api/documents/{doc_id}` | Remove a specific document |
| `DELETE` | `/api/documents` | Clear all documents |
| `GET` | `/api/health` | Health check (document/chunk counts) |
| `GET` | `/` | Chat UI |

### Query example

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```

Response includes the answer, detected intent, citations with relevance scores, the transformed query, and any warnings (e.g., if hallucinated content was filtered).

## Design Decisions

**Recursive chunking over fixed-size windows** — Fixed-size chunking cuts text at arbitrary positions, often splitting sentences or paragraphs mid-thought. Recursive splitting tries paragraph boundaries first, then sentence boundaries, then word boundaries. This produces chunks that are semantically coherent, which improves retrieval quality.

**BM25 + semantic hybrid over pure semantic search** — Embeddings are great at capturing meaning but can miss exact keyword matches. If a document mentions "TKA" (total knee arthroplasty) and the user searches for "TKA", a pure semantic search might rank it lower than a passage about "joint replacement." BM25 catches these exact matches. Combining both gives the best of both worlds.

**RRF over linear score interpolation** — BM25 scores and cosine similarity scores are on completely different scales (BM25 can be 0-20+, cosine is 0-1). Naively doing `0.7 * cosine + 0.3 * bm25` would be dominated by whichever method produces larger numbers. RRF sidesteps this entirely by using rank positions: `score = 1/(k + rank)`. It's simple, robust, and has been shown to outperform linear combination in practice.

**In-memory vector store** — No external database dependency. Numpy handles the linear algebra efficiently for demo-scale workloads (thousands of chunks). For production with 100K+ documents, you'd swap in FAISS or a dedicated vector database.

**Similarity threshold for citations** — Rather than always returning an answer, the system checks if retrieved chunks are actually relevant (cosine similarity ≥ 0.35). If nothing passes the threshold, it says "insufficient evidence" instead of generating a potentially hallucinated answer.

## Bonus Features

- **Citations with similarity threshold** — Each answer includes source citations with filename, page number, and relevance score. Chunks below 0.35 cosine similarity are excluded
- **Answer shaping** — Detects if the user wants a list, comparison, or general answer and switches prompt templates accordingly
- **Hallucination filter** — Post-generation evidence check: a second LLM call verifies each sentence against the source material and strips unsupported claims
- **PII refusal** — Regex-based detection of SSNs, credit card numbers, and passport-like patterns. Queries containing PII are refused before reaching the LLM
- **Intent detection** — Greetings and chitchat are handled without unnecessary KB searches

## Libraries Used

| Library | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `httpx` | Async HTTP client for Mistral API |
| `pymupdf` | PDF text extraction |
| `numpy` | Vector math (cosine similarity, BM25 scoring) |
| `python-multipart` | File upload handling |
| `python-dotenv` | Environment variable management |
| `jinja2` | Template engine (FastAPI dependency) |

No LangChain. No LlamaIndex. No external vector database. All search and retrieval logic is built from scratch.

## Production Improvements

If scaling this beyond a demo, I would add:

- **FAISS or pgvector** for vector search at scale (in-memory numpy won't handle millions of chunks)
- **Persistent storage** with PostgreSQL so documents survive server restarts
- **Streaming responses** via Server-Sent Events for better UX on longer answers
- **Authentication** to protect the API and separate user document spaces
- **More file types** — DOCX, TXT, HTML, markdown
- **Chunk deduplication** across documents to avoid redundant retrieval
- **Configurable models** — let users pick between Mistral models based on speed/quality tradeoff

## Project Structure