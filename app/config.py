"""
Configuration for the RAG pipeline.
All settings centralized here for easy modification.
"""
import os
from pathlib import Path
 
from dotenv import load_dotenv
 
# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")
 
 
# --- Mistral AI ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not set. "
        "Create a .env file with MISTRAL_API_KEY=your_key or export it."
    )
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
MISTRAL_EMBED_MODEL = "mistral-embed"
MISTRAL_CHAT_MODEL = "mistral-small-latest"
EMBEDDING_DIMENSION = 1024  # mistral-embed output dimension
 
# --- Chunking ---
CHUNK_SIZE = 512          # target tokens per chunk
CHUNK_OVERLAP = 64        # overlapping tokens between chunks
MIN_CHUNK_LENGTH = 50     # discard chunks shorter than this (chars)
 
# --- Search ---
TOP_K = 10                # how many chunks to retrieve per method
FINAL_TOP_K = 5           # how many chunks after re-ranking to feed to LLM
SIMILARITY_THRESHOLD = 0.35  # minimum cosine similarity to consider relevant
BM25_WEIGHT = 0.35        # weight for BM25 in hybrid score
SEMANTIC_WEIGHT = 0.65    # weight for semantic in hybrid score
 
# --- Generation ---
MAX_GENERATION_TOKENS = 2048
TEMPERATURE = 0.2
 
# --- Upload limits ---
MAX_FILE_SIZE_MB = 50
MAX_FILES_PER_UPLOAD = 20
ALLOWED_EXTENSIONS = {".pdf"}
 
# --- PII patterns (for query refusal) ---
PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",            # SSN
    r"\b\d{16}\b",                         # credit card (no spaces)
    r"\b\d{4}[\s-]\d{4}[\s-]\d{4}[\s-]\d{4}\b",  # credit card (spaces/dashes)
    r"\b[A-Z]{1,2}\d{6,9}\b",             # passport-like
]