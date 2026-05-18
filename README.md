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