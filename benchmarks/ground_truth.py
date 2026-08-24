"""
benchmarks/ground_truth.py
Exact brute-force k-NN, used as the ground-truth oracle for Recall@K.
Vectorized (single distance-matrix computation), not FAISS/ANN.
"""
from __future__ import annotations
import numpy as np
from core.distance import Metric, batch_distance


def brute_force_knn(vectors: np.ndarray, queries: np.ndarray, k: int, metric: Metric | str = Metric.COSINE):
    """Returns an (n_queries, k) int array of ground-truth neighbor indices."""
    results = np.zeros((queries.shape[0], k), dtype=np.int64)
    for i, q in enumerate(queries):
        dists = batch_distance(q, vectors, metric)
        results[i] = np.argsort(dists)[:k]
    return results


def recall_at_k(predicted: list[int], ground_truth: np.ndarray) -> float:
    """Fraction of ground-truth neighbors present in the predicted set."""
    gt_set = set(ground_truth.tolist())
    if not gt_set:
        return 1.0
    hit = len(gt_set.intersection(predicted))
    return hit / len(gt_set)
