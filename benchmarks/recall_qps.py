"""
benchmarks/recall_qps.py
Sweeps efSearch (and optionally M) values, measures Recall@K vs Queries Per
Second against the exact brute-force baseline, and prints/saves a report
that can be plotted directly into the README.

Run:
    python -m benchmarks.recall_qps
"""
from __future__ import annotations
import csv
import time
import argparse

import numpy as np

from core.index import HNSWIndex
from core.distance import Metric
from benchmarks.generate_data import generate_clustered_vectors, generate_queries
from benchmarks.ground_truth import brute_force_knn, recall_at_k


def run_sweep(
    n_vectors: int = 5000,
    dim: int = 64,
    n_queries: int = 200,
    k: int = 10,
    metric: str = "cosine",
    m_values: list[int] = (8, 16, 32),
    ef_search_values: list[int] = (10, 20, 50, 100, 200),
    ef_construction: int = 200,
    out_csv: str = "benchmarks/results.csv",
):
    vectors, _ = generate_clustered_vectors(n_vectors, dim)
    queries = generate_queries(n_queries, dim)

    print(f"Computing exact ground truth for {n_queries} queries over {n_vectors} vectors...")
    gt = brute_force_knn(vectors, queries, k, metric=metric)

    rows = []
    for M in m_values:
        print(f"\n== Building index with M={M}, efConstruction={ef_construction} ==")
        index = HNSWIndex(dim=dim, metric=metric, M=M, ef_construction=ef_construction, max_elements=n_vectors + 10)
        t0 = time.perf_counter()
        for v in vectors:
            index.insert(v)
        build_time = time.perf_counter() - t0
        print(f"  build_time={build_time:.2f}s ({n_vectors / build_time:.1f} inserts/s)")

        for ef in ef_search_values:
            t0 = time.perf_counter()
            recalls = []
            for i, q in enumerate(queries):
                hits = index.search(q, k=k, ef_search=ef)
                predicted_ids = [h[0] for h in hits]
                recalls.append(recall_at_k(predicted_ids, gt[i]))
            elapsed = time.perf_counter() - t0
            qps = n_queries / elapsed
            mean_recall = float(np.mean(recalls))
            print(f"  M={M:>3} ef={ef:>4}  recall@{k}={mean_recall:.4f}  qps={qps:8.1f}")
            rows.append({
                "M": M, "ef_search": ef, "recall_at_k": mean_recall,
                "qps": qps, "n_vectors": n_vectors, "dim": dim, "k": k,
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} data points to {out_csv}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_vectors", type=int, default=5000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n_queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    run_sweep(n_vectors=args.n_vectors, dim=args.dim, n_queries=args.n_queries, k=args.k)
