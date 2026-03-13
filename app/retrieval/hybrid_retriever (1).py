"""
Hybrid Retriever — BM25 + E5 with Weighted Score Fusion
=========================================================
Core contribution of the thesis. Combines sparse and dense retrieval
via min-max normalized weighted score fusion.

Key finding: optimal α = 0.70 (not the assumed default of 0.5).
Discovered via systematic grid search across α ∈ [0.0, 1.0] on 100+ MS MARCO queries.

    score_hybrid(d) = α · score_E5_norm(d) + (1 - α) · score_BM25_norm(d)

References
----------
- Robertson & Zaragoza (2009) — BM25 probabilistic framework
- Wang et al. (2024) — E5 text embedding model
- Oche et al. (2025) — RAG survey identifying this gap (arXiv:2507.18910)
"""

import time
import logging
from typing import List, Dict, Tuple, Optional

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.cache import MultiLevelCache
from app.retrieval.normalizer import min_max_normalize

logger = logging.getLogger(__name__)


class RetrievalResult:
    """Structured result from hybrid retrieval."""

    __slots__ = ("doc_id", "text", "score", "rank", "bm25_score", "dense_score")

    def __init__(
        self,
        doc_id: int,
        text: str,
        score: float,
        rank: int,
        bm25_score: float = 0.0,
        dense_score: float = 0.0,
    ):
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.rank = rank
        self.bm25_score = bm25_score
        self.dense_score = dense_score

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": round(self.score, 4),
            "rank": self.rank,
            "bm25_score": round(self.bm25_score, 4),
            "dense_score": round(self.dense_score, 4),
        }


class HybridRetriever:
    """
    Production-ready hybrid retriever with multi-level caching.

    Five-stage pipeline
    -------------------
    Stage 1 (380ms) — BM25 sparse retrieval
    Stage 2 (75ms)  — E5 query embedding
    Stage 3 (220ms) — FAISS dense search
    Stage 4 (35ms)  — Min-max normalization + weighted fusion
    Stage 5         — Cache hit detection (bypasses stages 1-4)

    Total: ~710ms average (vs. 84,000ms naive prototype → 99.2% improvement)

    Parameters
    ----------
    bm25 : BM25Retriever
    dense : DenseRetriever
    passages : list[str]
        Passage text store (parallel to index positions).
    alpha : float
        Dense weight in fusion. 0.0 = BM25-only, 1.0 = dense-only.
        Empirically optimal: α=0.70 on MS MARCO (p<0.001).
    k : int
        Number of results to return.
    cache : MultiLevelCache, optional
        Shared cache instance. Pass None to disable caching.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        passages: List[str],
        alpha: float = 0.70,
        k: int = 10,
        cache: Optional[MultiLevelCache] = None,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"α must be in [0, 1], got {alpha}")
        self.bm25 = bm25
        self.dense = dense
        self.passages = passages
        self.alpha = alpha
        self.k = k
        self.cache = cache or MultiLevelCache()
        self._query_count = 0
        self._total_latency_ms = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
        method: str = "hybrid",
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k passages for a query.

        Parameters
        ----------
        query : str
        k : int, optional  Override default k.
        alpha : float, optional  Override default alpha (for interactive exploration).
        method : str  One of "hybrid", "sparse", "dense".

        Returns
        -------
        List of RetrievalResult sorted by score descending.
        """
        k = k or self.k
        alpha = alpha if alpha is not None else self.alpha
        t_start = time.perf_counter()

        # --- Cache check (retrieval level) ---
        cache_key = f"{method}:{alpha}:{k}:{query}"
        cached = self.cache.get_retrieval(cache_key)
        if cached is not None:
            logger.debug(f"Retrieval cache HIT: '{query[:40]}...'")
            return cached

        # --- Stage 1: BM25 sparse retrieval ---
        if method in ("hybrid", "sparse"):
            t0 = time.perf_counter()
            bm25_results: Dict[int, float] = dict(self.bm25.retrieve(query, k=k))
            logger.debug(f"BM25: {(time.perf_counter()-t0)*1000:.0f}ms, {len(bm25_results)} hits")
        else:
            bm25_results = {}

        # --- Stage 2+3: Query embedding + FAISS dense search ---
        if method in ("hybrid", "dense"):
            # Check embedding cache first
            cached_vec = self.cache.get_embedding(query)
            if cached_vec is None:
                t0 = time.perf_counter()
                query_vec = self.dense.encode_query(query)
                logger.debug(f"Embedding: {(time.perf_counter()-t0)*1000:.0f}ms")
                self.cache.set_embedding(query, query_vec)
            else:
                logger.debug("Embedding cache HIT")
                query_vec = cached_vec

            t0 = time.perf_counter()
            dense_results: Dict[int, float] = dict(self.dense.retrieve(query, k=k))
            logger.debug(f"FAISS: {(time.perf_counter()-t0)*1000:.0f}ms, {len(dense_results)} hits")
        else:
            dense_results = {}

        # --- Stage 4: Score normalization + weighted fusion ---
        results = self._fuse(bm25_results, dense_results, alpha, k)

        # Attach passage texts
        final = [
            RetrievalResult(
                doc_id=doc_id,
                text=self.passages[doc_id] if doc_id < len(self.passages) else "",
                score=score,
                rank=rank + 1,
                bm25_score=bm25_results.get(doc_id, 0.0),
                dense_score=dense_results.get(doc_id, 0.0),
            )
            for rank, (doc_id, score) in enumerate(results)
        ]

        # Store in retrieval cache
        self.cache.set_retrieval(cache_key, final)

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._query_count += 1
        self._total_latency_ms += elapsed_ms
        logger.info(f"Query complete in {elapsed_ms:.0f}ms (avg {self.avg_latency_ms:.0f}ms)")

        return final

    # ------------------------------------------------------------------
    # Fusion logic
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse(
        bm25_scores: Dict[int, float],
        dense_scores: Dict[int, float],
        alpha: float,
        k: int,
    ) -> List[Tuple[int, float]]:
        """
        Min-max normalize both score sets then compute weighted hybrid score.

        Why min-max normalization?
        BM25 scores are unbounded (0 to 50+, query-length dependent).
        E5 scores are cosine similarities in [-1, 1].
        Direct combination is meaningless without a shared scale.

        score_hybrid = α · score_dense_norm + (1-α) · score_bm25_norm
        """
        bm25_norm = min_max_normalize(bm25_scores)
        dense_norm = min_max_normalize(dense_scores)

        # Candidate pool = union of both result sets
        all_docs = set(bm25_norm) | set(dense_norm)

        hybrid_scores = {
            doc_id: (
                alpha * dense_norm.get(doc_id, 0.0)
                + (1 - alpha) * bm25_norm.get(doc_id, 0.0)
            )
            for doc_id in all_docs
        }

        # Re-rank by hybrid score
        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def avg_latency_ms(self) -> float:
        if self._query_count == 0:
            return 0.0
        return self._total_latency_ms / self._query_count

    def stats(self) -> dict:
        return {
            "queries_processed": self._query_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "cache_stats": self.cache.stats(),
            "alpha": self.alpha,
            "k": self.k,
        }

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(α={self.alpha}, k={self.k}, "
            f"passages={len(self.passages):,})"
        )
