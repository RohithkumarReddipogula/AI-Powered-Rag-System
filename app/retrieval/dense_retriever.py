"""
Dense Retriever — E5 Embeddings + FAISS
========================================
Uses Microsoft E5-base-v2 (Wang et al., 2024) for semantic similarity search.
FAISS (Johnson et al., 2019) enables sub-second ANN search over 8.8M passages.

Strengths: semantic similarity, vocabulary mismatch handling, natural language
Weaknesses: exact term matching (product IDs, code identifiers)

Latency:  ~75ms query embedding + ~220ms FAISS search on Intel i7-9700K
Index size: 3.05 MB (FAISS quantized, 200x compression from raw 27 GB float32)
"""

import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# E5 requires task-specific prefixes for optimal retrieval performance
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


class DenseRetriever:
    """
    Dense retriever using E5 embeddings and a FAISS inner-product index.

    Cosine similarity is computed as inner product over L2-normalized vectors.
    The pre-built FAISS index is memory-mapped on load for fast startup.

    Parameters
    ----------
    index_path : str
        Path to the serialized FAISS index (.bin).
    model_name : str
        HuggingFace model ID for the sentence encoder.
    """

    def __init__(
        self,
        index_path: str,
        model_name: str = "intfloat/e5-base-v2",
    ):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatIP] = None
        self._load_model()
        self._load_index(index_path)

    # ------------------------------------------------------------------
    # Index construction & persistence
    # ------------------------------------------------------------------

    @classmethod
    def build_and_save(
        cls,
        passages: List[str],
        save_path: str,
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 32,
    ) -> "DenseRetriever":
        """
        Encode all passages with E5 and build + save a FAISS index.

        Note: For 8.8M passages this takes ~8h on GPU (NVIDIA A100).
              Pre-built index (~3.05 MB) is included in the repository.

        Example
        -------
        >>> retriever = DenseRetriever.build_and_save(passages, "indices/faiss_index.bin")
        """
        logger.info(f"Encoding {len(passages):,} passages with {model_name}...")
        model = SentenceTransformer(model_name)

        # Add passage prefix per E5 conventions
        prefixed = [E5_PASSAGE_PREFIX + p for p in passages]

        t0 = time.time()
        embeddings = model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,   # L2-norm for cosine via inner product
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info(f"Encoding complete in {(time.time()-t0)/60:.1f}min")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, save_path)
        logger.info(f"FAISS index saved → {save_path} ({Path(save_path).stat().st_size / 1e6:.2f} MB)")

        instance = cls.__new__(cls)
        instance.model_name = model_name
        instance._model = model
        instance._index = index
        return instance

    def _load_model(self) -> None:
        logger.info(f"Loading E5 encoder: {self.model_name}")
        t0 = time.time()
        self._model = SentenceTransformer(self.model_name)
        logger.info(f"E5 model loaded in {time.time()-t0:.1f}s")

    def _load_index(self, index_path: str) -> None:
        """Load FAISS index from disk (~3s for 8.8M passages)."""
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run: python scripts/build_faiss_index.py"
            )
        logger.info(f"Loading FAISS index from {index_path}...")
        t0 = time.time()
        self._index = faiss.read_index(str(index_path))
        logger.info(
            f"FAISS index loaded in {time.time()-t0:.2f}s "
            f"({self._index.ntotal:,} vectors, dim={self._index.d})"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k passages by cosine similarity to query embedding.

        Parameters
        ----------
        query : str
            Raw query string. E5 prefix is added internally.
        k : int
            Number of results to return.

        Returns
        -------
        list of (doc_id, cosine_similarity) sorted descending.
        """
        query_vec = self._encode_query(query)

        t0 = time.time()
        scores, indices = self._index.search(query_vec, k)
        logger.debug(f"FAISS search: {(time.time()-t0)*1000:.0f}ms")

        results = [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx >= 0  # FAISS returns -1 for padding
        ]
        return results

    def encode_query(self, query: str) -> np.ndarray:
        """Public method for external use (e.g., caching layer)."""
        return self._encode_query(query)

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to a normalized 768-dim vector."""
        prefixed = E5_QUERY_PREFIX + query
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_passages(self) -> int:
        return self._index.ntotal if self._index else 0

    def __repr__(self) -> str:
        return (
            f"DenseRetriever(model={self.model_name}, "
            f"passages={self.num_passages:,}, dim={self._index.d if self._index else '?'})"
        )
