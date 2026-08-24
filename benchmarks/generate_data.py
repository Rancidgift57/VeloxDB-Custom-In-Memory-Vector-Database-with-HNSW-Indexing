"""
benchmarks/generate_data.py
Generates synthetic clustered embeddings (more realistic than pure uniform
random -- real embeddings cluster around semantic centroids) for benchmarking.
"""
from __future__ import annotations
import numpy as np


def generate_clustered_vectors(n: int, dim: int, n_clusters: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(n_clusters, dim)).astype(np.float32)
    assignments = rng.integers(0, n_clusters, size=n)
    noise = rng.normal(0, 0.15, size=(n, dim)).astype(np.float32)
    vectors = centroids[assignments] + noise
    return vectors.astype(np.float32), assignments


def generate_queries(n_queries: int, dim: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n_queries, dim)).astype(np.float32)
