"""
tests/test_basic.py
Run with: pytest tests/ -v
"""
import numpy as np
import pytest

from core.index import HNSWIndex
from core.distance import Metric, cosine_distance, euclidean_distance
from filtering.metadata_filter import compile_filter
from benchmarks.generate_data import generate_clustered_vectors
from benchmarks.ground_truth import brute_force_knn, recall_at_k


def test_distance_shapes():
    a = np.random.rand(16).astype(np.float32)
    b = np.random.rand(50, 16).astype(np.float32)
    assert cosine_distance(a, b).shape == (50,)
    assert euclidean_distance(a, b).shape == (50,)


def test_insert_and_search_returns_self():
    idx = HNSWIndex(dim=8, metric=Metric.EUCLIDEAN, M=8, ef_construction=50)
    vecs = np.random.rand(100, 8).astype(np.float32)
    for v in vecs:
        idx.insert(v)
    query = vecs[42]
    results = idx.search(query, k=1, ef_search=50)
    assert results[0][0] == 42
    assert results[0][1] == pytest.approx(0.0, abs=1e-4)


def test_recall_is_reasonably_high():
    vectors, _ = generate_clustered_vectors(1000, 32, seed=3)
    idx = HNSWIndex(dim=32, metric=Metric.COSINE, M=16, ef_construction=100)
    for v in vectors:
        idx.insert(v)
    queries = vectors[:50]
    gt = brute_force_knn(vectors, queries, k=10, metric=Metric.COSINE)
    recalls = []
    for i, q in enumerate(queries):
        hits = idx.search(q, k=10, ef_search=100)
        recalls.append(recall_at_k([h[0] for h in hits], gt[i]))
    assert np.mean(recalls) > 0.85


def test_tombstone_delete_excludes_from_results():
    idx = HNSWIndex(dim=4, M=8, ef_construction=50)
    ids = [idx.insert(np.random.rand(4).astype(np.float32)) for _ in range(30)]
    target = ids[5]
    assert idx.delete(target) is True
    query = idx.vectors[target]
    results = idx.search(query, k=30, ef_search=50)
    returned_ids = [r[0] for r in results]
    assert target not in returned_ids
    assert idx.delete(target) is False  # already deleted


def test_compact_purges_tombstones():
    idx = HNSWIndex(dim=4, M=8, ef_construction=50)
    ids = [idx.insert(np.random.rand(4).astype(np.float32)) for _ in range(20)]
    for nid in ids[:5]:
        idx.delete(nid)
    assert idx.stats()["tombstoned"] == 5
    idx.compact(background=False)  # run synchronously so we can assert right after
    assert idx.stats()["tombstoned"] == 0
    assert idx.stats()["live_nodes"] == 15


def test_compact_runs_in_background_without_blocking():
    idx = HNSWIndex(dim=4, M=8, ef_construction=50)
    ids = [idx.insert(np.random.rand(4).astype(np.float32)) for _ in range(200)]
    for nid in ids[:50]:
        idx.delete(nid)

    thread = idx.compact(background=True)  # default; returns immediately
    assert thread is not None
    # index must remain fully usable *while* the rebuild is in flight
    query = np.random.rand(4).astype(np.float32)
    results = idx.search(query, k=5, ef_search=50)
    assert len(results) > 0
    extra_id = idx.insert(np.random.rand(4).astype(np.float32))
    assert extra_id is not None

    thread.join(timeout=10)
    assert idx.stats()["tombstoned"] == 0


def test_metadata_filter_dsl():
    f = compile_filter({"category": "finance", "score": {"$gte": 5}})
    assert f({"category": "finance", "score": 10}) is True
    assert f({"category": "finance", "score": 2}) is False
    assert f({"category": "tech", "score": 10}) is False


def test_hybrid_filtered_search():
    idx = HNSWIndex(dim=4, M=8, ef_construction=50)
    for i in range(50):
        cat = "finance" if i % 2 == 0 else "tech"
        idx.insert(np.random.rand(4).astype(np.float32), metadata={"category": cat})

    f = compile_filter({"category": "finance"})
    query = np.random.rand(4).astype(np.float32)
    results = idx.search(query, k=10, ef_search=50, filter_fn=f)
    assert len(results) > 0
    assert all(meta["category"] == "finance" for _, _, meta in results)


def test_concurrent_inserts_and_searches_dont_corrupt_graph():
    """Fine-grained (per-node) locking test: many threads inserting and
    searching simultaneously should never crash, never lose/corrupt an
    edge list, and every inserted vector should remain findable afterward."""
    import threading

    idx = HNSWIndex(dim=8, M=8, ef_construction=50, max_elements=2000)
    n_threads = 8
    inserts_per_thread = 30
    all_inserted_ids = []
    ids_lock = threading.Lock()
    errors = []

    def writer(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(inserts_per_thread):
                vec = rng.random(8).astype(np.float32)
                nid = idx.insert(vec)
                with ids_lock:
                    all_inserted_ids.append(nid)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(50):
                q = rng.random(8).astype(np.float32)
                idx.search(q, k=5, ef_search=30)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(n_threads)]
    threads += [threading.Thread(target=reader, args=(i + 100,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"concurrent access raised: {errors}"
    assert len(all_inserted_ids) == n_threads * inserts_per_thread
    assert len(set(all_inserted_ids)) == len(all_inserted_ids)  # no duplicate/racy ids

    # every inserted vector should still be reachable via search
    found = 0
    for nid in all_inserted_ids[:40]:
        vec = idx.vectors[nid]
        hits = idx.search(vec, k=1, ef_search=100)
        if hits and hits[0][0] == nid:
            found += 1
    assert found >= 36  # allow a small ANN miss margin, not exact-100%


def test_quantized_index_end_to_end():
    """int8 storage: fit -> insert -> search -> compact should all work,
    and recall should stay high despite the ~4x memory-reduced storage."""
    from benchmarks.generate_data import generate_clustered_vectors
    from benchmarks.ground_truth import brute_force_knn, recall_at_k

    vectors, _ = generate_clustered_vectors(500, 16, seed=7)

    idx = HNSWIndex(dim=16, metric="euclidean", M=8, ef_construction=100, quantize=True)
    # storage dtype should actually be uint8, proving the 4x reduction is real
    assert idx.vectors.dtype == np.uint8

    with pytest.raises(RuntimeError):
        idx.insert(vectors[0])  # must fit_quantizer() first

    idx.fit_quantizer(vectors[:100])
    for v in vectors:
        idx.insert(v)

    queries = vectors[:30]
    gt = brute_force_knn(vectors, queries, k=5, metric="euclidean")
    recalls = []
    for i, q in enumerate(queries):
        hits = idx.search(q, k=5, ef_search=100)
        recalls.append(recall_at_k([h[0] for h in hits], gt[i]))
    mean_recall = float(np.mean(recalls))
    assert mean_recall > 0.6, f"quantized recall too low: {mean_recall}"

    # delete + synchronous compact must survive the quantized re-encode path
    idx.delete(0)
    idx.compact(background=False)
    assert idx.vectors.dtype == np.uint8
    assert idx.stats()["tombstoned"] == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
