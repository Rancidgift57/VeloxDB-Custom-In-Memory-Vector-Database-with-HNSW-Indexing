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
    def __init__(self, dim: int, margin: float = 0.10):
        """`margin`: fraction of the fitted [min, max] per-dimension range
        to pad on *each* side at fit time (default 10%). Encoding is
        clip-at-the-edges by construction (float -> uint8 has to clamp
        somewhere), so a sample that doesn't perfectly cover the true
        data distribution means later, slightly-out-of-sample vectors
        silently clip to the boundary and lose precision. Padding the
        fitted range gives real headroom for that without requiring a
        perfectly representative fit sample -- it costs a small amount of
        resolution (same 256 buckets now cover a wider span) in exchange
        for tolerating drift. Set margin=0.0 to reproduce the old
        tight-fit behavior exactly."""
        self.dim = dim
        self.margin = margin
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.fitted = False
        # Clipping telemetry: how many scalar components (not whole
        # vectors) have hit the [0, 255] boundary during encode, out of
        # how many encoded total. A rising clip rate is the actionable
        # signal that the fitted range has drifted and a re-fit (e.g. via
        # compact()) is due -- see should_refit().
        self._clipped_components = 0
        self._total_components = 0

    def fit(self, sample: np.ndarray) -> None:
        """sample: (n, dim) representative float32 vectors."""
        raw_min = sample.min(axis=0).astype(np.float32)
        raw_max = sample.max(axis=0).astype(np.float32)
        pad = (raw_max - raw_min) * self.margin
        self.min_vals = raw_min - pad
        self.max_vals = raw_max + pad
        rng = np.maximum(self.max_vals - self.min_vals, 1e-8)
        self.scale = (rng / 255.0).astype(np.float32)
        self.fitted = True
        self._clipped_components = 0
        self._total_components = 0

    def _track_clipping(self, q_raw: np.ndarray) -> None:
        # q_raw is the pre-clip quantized value(s); anything outside
        # [0, 255] would have been clamped, i.e. lost precision.
        clipped = np.count_nonzero((q_raw < 0) | (q_raw > 255))
        self._clipped_components += int(clipped)
        self._total_components += int(q_raw.size)

    def clip_stats(self) -> dict:
        """Fraction of encoded scalar components that have hit the
        quantization boundary since the last fit(). A near-zero rate is
        healthy; a rate that keeps climbing means the live data has
        drifted outside the fitted range and it's time to re-fit
        (automatic on compact(), or call fit() again manually)."""
        rate = (
            self._clipped_components / self._total_components
            if self._total_components else 0.0
        )
        return {
            "clipped": self._clipped_components,
            "total": self._total_components,
            "clip_rate": rate,
        }

    def should_refit(self, threshold: float = 0.02, min_samples: int = 200) -> bool:
        """Convenience check: True once enough components have been
        encoded to be statistically meaningful AND the clip rate exceeds
        `threshold` (default 2%). Doesn't refit automatically -- encode()
        is called per-insert and re-fitting requires re-encoding the
        whole live corpus, which is a compact()-sized operation -- this
        just gives callers/monitoring an honest signal for when to
        trigger one."""
        stats = self.clip_stats()
        return stats["total"] >= min_samples and stats["clip_rate"] > threshold

    def encode(self, vec: np.ndarray) -> np.ndarray:
        """float32 (dim,) -> uint8 (dim,)"""
        assert self.fitted, "call fit() before encode()"
        q = (vec - self.min_vals) / self.scale
        self._track_clipping(np.round(q))
        return np.clip(np.round(q), 0, 255).astype(np.uint8)

    def encode_batch(self, mat: np.ndarray) -> np.ndarray:
        assert self.fitted, "call fit() before encode_batch()"
        q = (mat - self.min_vals) / self.scale
        self._track_clipping(np.round(q))
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
