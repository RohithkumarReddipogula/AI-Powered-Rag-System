"""
Score Normalization Utilities
==============================
Min-max normalization for hybrid score fusion.
Maps heterogeneous score ranges to a common [0, 1] scale.

BM25:  raw scores ∈ [0, 50+]  (query-length dependent)
E5:    cosine similarities ∈ [-1, 1]  (typically [0.6, 0.95] for relevant docs)

Without normalization, BM25's higher magnitude would dominate the fusion.
"""

from typing import Dict


def min_max_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Apply min-max normalization to a dict of {doc_id: score}.

    Maps [min_score, max_score] → [0, 1].
    Returns empty dict for empty or single-element input.

    Edge cases
    ----------
    - Empty input: returns {}
    - All scores identical: returns all-zero dict (avoids division by zero)
    """
    if not scores:
        return {}

    values = list(scores.values())
    min_s = min(values)
    max_s = max(values)
    denom = max_s - min_s

    if denom == 0:
        # All scores identical — can't distinguish; map to 0
        return {doc_id: 0.0 for doc_id in scores}

    return {
        doc_id: (score - min_s) / denom
        for doc_id, score in scores.items()
    }


def standard_normalize(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Z-score normalization. More robust to outliers than min-max,
    but requires the scores to be approximately normally distributed.
    Provided as an alternative for future experiments.
    """
    if not scores:
        return {}

    values = list(scores.values())
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5

    if std == 0:
        return {doc_id: 0.0 for doc_id in scores}

    return {
        doc_id: (score - mean) / std
        for doc_id, score in scores.items()
    }
