"""
core/distance.py
Vectorized distance / similarity kernels used by the HNSW index.

All functions accept:
    a: np.ndarray of shape (d,)            -- a single query vector
    b: np.ndarray of shape (n, d)          -- a batch of candidate vectors
and return:
    np.ndarray of shape (n,)               -- distance from a to every row of b

Keeping a single-vector-vs-batch signature lets the HNSW graph traversal
evaluate a whole candidate neighbor list in one BLAS call instead of a
Python for-loop, which is where most naive HNSW implementations lose
their vectorization advantage.
"""
from __future__ import annotations
import numpy as np
from enum import Enum


class Metric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared-root L2 distance, vectorized via matrix norms (no python loop)."""
    diff = b - a  # broadcasting (n, d) - (d,)
    # np.einsum is faster than np.linalg.norm for this pattern
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 - cosine_similarity, vectorized."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = b_norm @ a_norm
    return 1.0 - sim


def dot_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Negative dot product so 'lower is better' holds like the other metrics."""
    return -(b @ a)


_DISPATCH = {
    Metric.COSINE: cosine_distance,
    Metric.EUCLIDEAN: euclidean_distance,
    Metric.DOT: dot_distance,
}


def batch_distance(a: np.ndarray, b: np.ndarray, metric: Metric | str) -> np.ndarray:
    metric = Metric(metric)
    return _DISPATCH[metric](a, b)


def pairwise_distance_matrix(x: np.ndarray, metric: Metric | str) -> np.ndarray:
    """Full N x N distance matrix, used only by the brute-force ground-truth
    baseline in benchmarks/ -- never called on the hot query path."""
    metric = Metric(metric)
    if metric == Metric.EUCLIDEAN:
        sq = np.sum(x * x, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2 * (x @ x.T)
        np.maximum(d2, 0, out=d2)
        return np.sqrt(d2)
    elif metric == Metric.COSINE:
        norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return 1.0 - (norm @ norm.T)
    else:
        return -(x @ x.T)
