"""
core/quantization.py
Scalar (per-dimension, min/max) quantization: float32 -> int8.

Reduces raw vector RAM footprint ~4x (32 bits -> 8 bits per component).
Quantization params (min/max per dimension) are fit once on a representative
sample/corpus, then applied to every insert. Distances are computed by
dequantizing back to float32 (cheap, vectorized) rather than doing integer
distance math, which keeps this a drop-in accuracy/memory tradeoff rather
than a separate code path.
"""
from __future__ import annotations
import numpy as np


class ScalarQuantizer:
    def __init__(self, dim: int):
        self.dim = dim
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.fitted = False

    def fit(self, sample: np.ndarray) -> None:
        """sample: (n, dim) representative float32 vectors."""
        self.min_vals = sample.min(axis=0).astype(np.float32)
        self.max_vals = sample.max(axis=0).astype(np.float32)
        rng = np.maximum(self.max_vals - self.min_vals, 1e-8)
        self.scale = (rng / 255.0).astype(np.float32)
        self.fitted = True

    def encode(self, vec: np.ndarray) -> np.ndarray:
        """float32 (dim,) -> uint8 (dim,)"""
        assert self.fitted, "call fit() before encode()"
        q = (vec - self.min_vals) / self.scale
        return np.clip(np.round(q), 0, 255).astype(np.uint8)

    def encode_batch(self, mat: np.ndarray) -> np.ndarray:
        assert self.fitted, "call fit() before encode_batch()"
        q = (mat - self.min_vals) / self.scale
        return np.clip(np.round(q), 0, 255).astype(np.uint8)

    def decode(self, qvec: np.ndarray) -> np.ndarray:
        """uint8 (dim,) -> float32 (dim,) reconstruction."""
        assert self.fitted, "call fit() before decode()"
        return (qvec.astype(np.float32) * self.scale) + self.min_vals

    def decode_batch(self, qmat: np.ndarray) -> np.ndarray:
        assert self.fitted, "call fit() before decode_batch()"
        return (qmat.astype(np.float32) * self.scale) + self.min_vals

    def memory_savings(self, n_vectors: int) -> dict:
        f32_bytes = n_vectors * self.dim * 4
        i8_bytes = n_vectors * self.dim * 1
        return {
            "float32_mb": f32_bytes / 1e6,
            "int8_mb": i8_bytes / 1e6,
            "reduction_factor": f32_bytes / i8_bytes,
        }
