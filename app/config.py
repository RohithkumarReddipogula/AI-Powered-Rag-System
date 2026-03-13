"""
Central configuration for the AI-Powered RAG System.
All tunable parameters in one place for reproducibility.
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalConfig:
    # Hybrid fusion — empirically optimized via grid search (see scripts/parameter_optimization.py)
    # Default α=0.70 outperforms the commonly assumed α=0.5 (p<0.001, t(99)=3.14)
    alpha: float = 0.70

    # Number of results to retrieve
    k: int = 10

    # BM25 hyperparameters (Okapi BM25 standard values)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # E5 embedding model
    embedding_model: str = "intfloat/e5-base-v2"
    embedding_dim: int = 768
    embedding_batch_size: int = 32

    # FAISS index type (flat = exact search, IVF = approximate)
    faiss_index_type: str = "flat"


@dataclass
class CacheConfig:
    # Three-level caching strategy — achieves 40-45% hit rate (Zipfian query distribution)
    retrieval_cache_size: int = 10_000    # ~50 MB memory
    embedding_cache_size: int = 50_000   # ~150 MB memory
    generation_cache_size: int = 1_000   # ~20 MB memory


@dataclass
class GenerationConfig:
    model: str = "gpt-4"
    max_tokens: int = 512
    temperature: float = 0.1
    # k for generation context (fewer passages = cheaper; k=10 is used for retrieval eval)
    context_k: int = 5


@dataclass
class PathConfig:
    bm25_index: str = "indices/bm25_index.pkl"
    faiss_index: str = "indices/faiss_index.bin"
    passage_store: str = "indices/passages.jsonl"


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"


@dataclass
class AppConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))


# Global singleton
config = AppConfig()
