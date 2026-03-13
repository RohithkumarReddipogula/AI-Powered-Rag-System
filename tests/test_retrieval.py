"""
Unit tests for the hybrid retrieval pipeline.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch

from app.retrieval.normalizer import min_max_normalize, standard_normalize
from app.retrieval.cache import LRUCache, MultiLevelCache
from app.retrieval.hybrid_retriever import HybridRetriever
from app.evaluation.metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank


# ─── Normalizer tests ─────────────────────────────────────────────────────────

class TestMinMaxNormalize:
    def test_basic(self):
        scores = {0: 10.0, 1: 20.0, 2: 30.0}
        result = min_max_normalize(scores)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_empty(self):
        assert min_max_normalize({}) == {}

    def test_all_identical(self):
        scores = {0: 5.0, 1: 5.0, 2: 5.0}
        result = min_max_normalize(scores)
        assert all(v == 0.0 for v in result.values())

    def test_single_entry(self):
        result = min_max_normalize({42: 7.5})
        assert result[42] == pytest.approx(0.0)

    def test_output_in_unit_range(self):
        import random
        scores = {i: random.uniform(0, 100) for i in range(50)}
        result = min_max_normalize(scores)
        assert all(0.0 <= v <= 1.0 for v in result.values())


# ─── Cache tests ──────────────────────────────────────────────────────────────

class TestLRUCache:
    def test_set_get(self):
        cache = LRUCache(maxsize=3)
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_miss_returns_none(self):
        cache = LRUCache(maxsize=3)
        assert cache.get("nonexistent") is None

    def test_eviction(self):
        cache = LRUCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # "a" should be evicted
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_hit_rate(self):
        cache = LRUCache(maxsize=10)
        cache.set("x", 100)
        cache.get("x")   # hit
        cache.get("y")   # miss
        assert cache.hit_rate == pytest.approx(0.5)

    def test_lru_order(self):
        """Accessing a key should prevent it from being evicted."""
        cache = LRUCache(maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")   # "a" is now most recently used
        cache.set("c", 3)  # "b" should be evicted, not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None


class TestMultiLevelCache:
    def test_retrieval_roundtrip(self):
        cache = MultiLevelCache()
        cache.set_retrieval("key1", ["result1", "result2"])
        assert cache.get_retrieval("key1") == ["result1", "result2"]

    def test_generation_key_stability(self):
        """Same query+passages should produce the same generation cache key."""
        cache = MultiLevelCache()
        passages = ["passage one", "passage two"]
        cache.set_generation("query", passages, "answer")
        assert cache.get_generation("query", passages) == "answer"

    def test_generation_key_sensitivity(self):
        """Different passages should produce a different cache key."""
        cache = MultiLevelCache()
        cache.set_generation("query", ["a"], "answer A")
        assert cache.get_generation("query", ["b"]) is None

    def test_stats_structure(self):
        cache = MultiLevelCache()
        stats = cache.stats()
        assert "retrieval" in stats
        assert "embedding" in stats
        assert "generation" in stats
        assert "overall_hit_rate" in stats


# ─── Evaluation metrics tests ─────────────────────────────────────────────────

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k([1, 2, 3], relevant_ids={2}, k=10) == 1.0

    def test_zero_recall(self):
        assert recall_at_k([1, 2, 3], relevant_ids={99}, k=10) == 0.0

    def test_cutoff_respected(self):
        # Relevant doc is at position 5, but k=3
        assert recall_at_k([1, 2, 3, 4, 5], relevant_ids={5}, k=3) == 0.0
        assert recall_at_k([1, 2, 3, 4, 5], relevant_ids={5}, k=5) == 1.0

    def test_empty_relevant(self):
        assert recall_at_k([1, 2, 3], relevant_ids=set(), k=10) == 0.0

    def test_multiple_relevant(self):
        result = recall_at_k([1, 2, 3], relevant_ids={1, 3, 99}, k=10)
        assert result == pytest.approx(2 / 3)


class TestMRR:
    def test_first_rank(self):
        assert reciprocal_rank([5, 2, 8], relevant_ids={5}) == pytest.approx(1.0)

    def test_second_rank(self):
        assert reciprocal_rank([5, 2, 8], relevant_ids={2}) == pytest.approx(0.5)

    def test_no_hit(self):
        assert reciprocal_rank([5, 2, 8], relevant_ids={99}) == 0.0

    def test_mrr_perfect(self):
        queries = [
            {"retrieved_ids": [1, 2, 3], "relevant_ids": {1}},
            {"retrieved_ids": [4, 5, 6], "relevant_ids": {4}},
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(1.0)

    def test_mrr_mixed(self):
        queries = [
            {"retrieved_ids": [1, 2], "relevant_ids": {1}},  # RR=1.0
            {"retrieved_ids": [1, 2], "relevant_ids": {2}},  # RR=0.5
        ]
        assert mean_reciprocal_rank(queries) == pytest.approx(0.75)

    def test_mrr_empty(self):
        assert mean_reciprocal_rank([]) == 0.0


# ─── Hybrid fusion tests (unit, mocked) ───────────────────────────────────────

class TestHybridFusion:
    """Test the _fuse static method independently from actual retrievers."""

    def test_alpha_zero_is_sparse_only(self):
        """At α=0.0, only BM25 scores should matter."""
        bm25 = {0: 10.0, 1: 5.0}
        dense = {2: 0.9}  # Should not appear at α=0.0

        results = HybridRetriever._fuse(bm25, dense, alpha=0.0, k=2)
        result_ids = [r[0] for r in results]
        assert 0 in result_ids  # BM25 top doc
        assert 2 not in result_ids  # Dense-only doc excluded

    def test_alpha_one_is_dense_only(self):
        """At α=1.0, only dense scores should matter."""
        bm25 = {0: 10.0}  # Should not affect at α=1.0
        dense = {1: 0.95, 2: 0.80}

        results = HybridRetriever._fuse(bm25, dense, alpha=1.0, k=2)
        result_ids = [r[0] for r in results]
        assert 1 in result_ids

    def test_output_length_capped_at_k(self):
        bm25 = {i: float(i) for i in range(20)}
        dense = {i: float(i) / 20 for i in range(20)}
        results = HybridRetriever._fuse(bm25, dense, alpha=0.5, k=5)
        assert len(results) == 5

    def test_union_of_result_sets(self):
        """Hybrid should consider docs from both retrievers."""
        bm25 = {0: 10.0}    # only in BM25
        dense = {1: 0.9}    # only in dense
        results = HybridRetriever._fuse(bm25, dense, alpha=0.5, k=5)
        result_ids = {r[0] for r in results}
        assert 0 in result_ids
        assert 1 in result_ids

    def test_scores_decrease_monotonically(self):
        """Results should be sorted by hybrid score descending."""
        bm25 = {i: float(10 - i) for i in range(5)}
        dense = {i: float(i) / 5 for i in range(5)}
        results = HybridRetriever._fuse(bm25, dense, alpha=0.5, k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)
