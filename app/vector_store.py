"""
In-memory vector store with both semantic (cosine) and keyword (BM25) search.
 
NO external vector database or search library is used — everything is built
from scratch with numpy for the linear algebra.
 
BM25 implementation notes:
──────────────────────────
BM25 (Best Matching 25) scores documents by term frequency, inverse document
frequency, and document length normalisation. Our implementation:
  - Tokenises on word boundaries, lowercased, with stopword removal.
  - Computes IDF as log((N - df + 0.5) / (df + 0.5) + 1) for stability.
  - Uses standard k1=1.5, b=0.75 parameters.
 
Hybrid search:
──────────────
We combine semantic and BM25 results using **Reciprocal Rank Fusion (RRF)**,
which is robust to score-scale differences between the two methods.
  hybrid_score = 1/(k + rank_semantic) + 1/(k + rank_bm25)
where k=60 is the standard smoothing constant.
"""
from __future__ import annotations
 
import logging
import math
import re
import threading
from collections import Counter, defaultdict
 
import numpy as np
 
from app.config import (
    BM25_WEIGHT,
    FINAL_TOP_K,
    SEMANTIC_WEIGHT,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.models import Chunk, SearchResult
 
logger = logging.getLogger(__name__)
 
# Common English stopwords for BM25
STOPWORDS = frozenset(
    "a an the is it in on at to for of and or but not with by from as this that "
    "be are was were been have has had do does did will would shall should can "
    "could may might i me my we our you your he she they them its".split()
)
 
 
def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation with stopword removal."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]
 
 
class VectorStore:
    """
    Thread-safe in-memory store for document chunks.
    Supports cosine similarity search and BM25 keyword search.
    """
 
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None  # shape (n, dim)
        self._doc_ids: set[str] = set()
 
        # BM25 index structures
        self._bm25_df: Counter = Counter()       # document frequency per term
        self._bm25_tf: list[Counter] = []         # term frequency per chunk
        self._bm25_dl: list[int] = []             # document (chunk) length
        self._bm25_avgdl: float = 0.0
        self._bm25_n: int = 0
 
    # ──────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────
 
    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add pre-embedded chunks to the store and rebuild BM25 index."""
        if not chunks:
            return
 
        with self._lock:
            new_embeddings = np.array(
                [c.embedding for c in chunks], dtype=np.float32
            )
 
            if self._embeddings is not None:
                self._embeddings = np.vstack([self._embeddings, new_embeddings])
            else:
                self._embeddings = new_embeddings
 
            self._chunks.extend(chunks)
            for c in chunks:
                self._doc_ids.add(c.doc_id)
 
            # Rebuild BM25 index (fast for <100k chunks)
            self._rebuild_bm25()
 
    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 statistics from all chunks."""
        self._bm25_df = Counter()
        self._bm25_tf = []
        self._bm25_dl = []
 
        for chunk in self._chunks:
            tokens = _tokenize(chunk.text)
            tf = Counter(tokens)
            self._bm25_tf.append(tf)
            self._bm25_dl.append(len(tokens))
            for term in set(tokens):
                self._bm25_df[term] += 1
 
        self._bm25_n = len(self._chunks)
        self._bm25_avgdl = (
            sum(self._bm25_dl) / self._bm25_n if self._bm25_n else 1.0
        )
 
    # ──────────────────────────────────────────
    # Semantic search
    # ──────────────────────────────────────────
 
    def semantic_search(
        self, query_embedding: list[float], top_k: int = TOP_K
    ) -> list[SearchResult]:
        """Cosine similarity search over stored embeddings."""
        with self._lock:
            if self._embeddings is None or len(self._chunks) == 0:
                return []
 
            q = np.array(query_embedding, dtype=np.float32)
            # Normalise
            q_norm = q / (np.linalg.norm(q) + 1e-10)
            e_norms = self._embeddings / (
                np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
            )
            scores = e_norms @ q_norm  # cosine similarities
 
            # Get top-k indices
            k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
 
            results = []
            for idx in top_indices:
                score = float(scores[idx])
                results.append(
                    SearchResult(
                        chunk=self._chunks[idx],
                        score=score,
                        method="semantic",
                    )
                )
            return results
 
    # ──────────────────────────────────────────
    # BM25 keyword search
    # ──────────────────────────────────────────
 
    def bm25_search(
        self, query: str, top_k: int = TOP_K, k1: float = 1.5, b: float = 0.75
    ) -> list[SearchResult]:
        """BM25 ranking over the stored chunks."""
        with self._lock:
            if not self._chunks:
                return []
 
            query_tokens = _tokenize(query)
            if not query_tokens:
                return []
 
            n = self._bm25_n
            scores = np.zeros(n, dtype=np.float64)
 
            for term in query_tokens:
                if term not in self._bm25_df:
                    continue
                df = self._bm25_df[term]
                # IDF with smoothing
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
 
                for i in range(n):
                    tf = self._bm25_tf[i].get(term, 0)
                    if tf == 0:
                        continue
                    dl = self._bm25_dl[i]
                    tf_norm = (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * dl / self._bm25_avgdl)
                    )
                    scores[i] += idf * tf_norm
 
            k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
 
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append(
                        SearchResult(
                            chunk=self._chunks[idx],
                            score=float(scores[idx]),
                            method="bm25",
                        )
                    )
            return results
 
    # ──────────────────────────────────────────
    # Hybrid search with RRF
    # ──────────────────────────────────────────
 
    def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = FINAL_TOP_K,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """
        Combine semantic and BM25 results using Reciprocal Rank Fusion.
 
        RRF is preferred over raw score interpolation because:
        - Semantic scores (cosine sim) and BM25 scores are on different scales.
        - RRF only uses rank positions, so it's naturally normalised.
        - It's been shown to outperform linear combination in practice.
 
        Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
        """
        semantic_results = self.semantic_search(query_embedding, TOP_K)
        bm25_results = self.bm25_search(query, TOP_K)
 
        # Build rank maps (chunk_id -> rank, 1-indexed)
        semantic_ranks: dict[str, int] = {}
        for rank, sr in enumerate(semantic_results, 1):
            semantic_ranks[sr.chunk.id] = rank
 
        bm25_ranks: dict[str, int] = {}
        for rank, sr in enumerate(bm25_results, 1):
            bm25_ranks[sr.chunk.id] = rank
 
        # Collect all unique chunks
        all_chunks: dict[str, SearchResult] = {}
        for sr in semantic_results + bm25_results:
            if sr.chunk.id not in all_chunks:
                all_chunks[sr.chunk.id] = sr
 
        # Compute RRF scores
        rrf_scores: list[tuple[str, float]] = []
        for cid in all_chunks:
            s_rank = semantic_ranks.get(cid, TOP_K + 50)  # penalty for missing
            b_rank = bm25_ranks.get(cid, TOP_K + 50)
            rrf = (
                SEMANTIC_WEIGHT / (rrf_k + s_rank)
                + BM25_WEIGHT / (rrf_k + b_rank)
            )
            rrf_scores.append((cid, rrf))
 
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
 
        # Build final results with original semantic score for threshold check
        final: list[SearchResult] = []
        for cid, rrf_score in rrf_scores[:top_k]:
            sr = all_chunks[cid]
            # Use the semantic score for threshold (more meaningful than RRF)
            sem_score = 0.0
            for sem_sr in semantic_results:
                if sem_sr.chunk.id == cid:
                    sem_score = sem_sr.score
                    break
            final.append(
                SearchResult(
                    chunk=sr.chunk,
                    score=sem_score,  # store semantic score for threshold
                    method="hybrid",
                )
            )
        return final
 
    # ──────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────
 
    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
 
    @property
    def num_documents(self) -> int:
        return len(self._doc_ids)
 
    def get_documents(self) -> list[dict]:
        """Return summary info about ingested documents."""
        doc_info: dict[str, dict] = {}
        for c in self._chunks:
            if c.doc_id not in doc_info:
                doc_info[c.doc_id] = {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "num_chunks": 0,
                    "pages": set(),
                }
            doc_info[c.doc_id]["num_chunks"] += 1
            doc_info[c.doc_id]["pages"].add(c.page_number)
        return [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "num_chunks": d["num_chunks"],
                "num_pages": len(d["pages"]),
            }
            for d in doc_info.values()
        ]
 
    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks for a document."""
        with self._lock:
            before = len(self._chunks)
            indices_to_keep = [
                i for i, c in enumerate(self._chunks) if c.doc_id != doc_id
            ]
            if len(indices_to_keep) == before:
                return False
 
            self._chunks = [self._chunks[i] for i in indices_to_keep]
            if self._embeddings is not None and indices_to_keep:
                self._embeddings = self._embeddings[indices_to_keep]
            elif not indices_to_keep:
                self._embeddings = None
 
            self._doc_ids.discard(doc_id)
            self._rebuild_bm25()
            return True
 
    def clear(self) -> None:
        """Remove all data."""
        with self._lock:
            self._chunks.clear()
            self._embeddings = None
            self._doc_ids.clear()
            self._bm25_df.clear()
            self._bm25_tf.clear()
            self._bm25_dl.clear()
            self._bm25_n = 0
            self._bm25_avgdl = 0.0
 
 
# Singleton instance
store = VectorStore()