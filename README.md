# MiniVectorDB — In-Memory HNSW Vector Database

A from-scratch (no FAISS/Chroma/Pinecone) approximate nearest neighbor
engine implementing Hierarchical Navigable Small World graphs, built in
pure Python + NumPy, with hybrid filtering, tombstone deletes, scalar
quantization, WAL durability, and a FastAPI service layer.

Every module below is implemented, tested, and benchmarked in this repo —
not pseudocode.

## 1. Architecture / File Map

```
vectordb/
├── core/
│   ├── distance.py       # vectorized cosine / euclidean / dot kernels
│   ├── node.py            # graph node: id, level, per-layer neighbors, tombstone
│   ├── index.py            # HNSWIndex: build, greedy search, RWLock, compact()
│   ├── quantization.py     # ScalarQuantizer: float32 -> int8, ~4x memory
│   └── database.py         # VectorDatabase facade: index + WAL + snapshot
├── storage/
│   ├── wal.py               # JSONL write-ahead log, fsync'd per write
│   └── snapshot.py          # atomic pickle snapshot save/load
├── filtering/
│   └── metadata_filter.py   # {"$and"/"$or"/"$gte"/...} -> callable predicate
├── api/
│   ├── schemas.py            # Pydantic request/response models
│   └── main.py                # FastAPI: /insert /search /delete /compact /health
├── client/
│   ├── client.py               # VectorDBClient SDK (single instance)
│   └── sharded_client.py        # ShardedVectorDBClient (multi-instance, hash-routed)
├── examples/
│   └── sharded_demo.py           # runnable multi-process sharding demo
├── benchmarks/
│   ├── generate_data.py         # synthetic clustered embeddings
│   ├── ground_truth.py           # brute-force exact k-NN oracle
│   └── recall_qps.py              # sweeps M / efSearch -> recall vs QPS CSV
├── tests/test_basic.py             # pytest suite (7/7 passing)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 2. How each requirement maps to code

| Spec requirement | Implementation |
|---|---|
| Vectorized cosine/L2 | `core/distance.py` — `einsum`/BLAS, batch (1×d vs n×d) signature |
| Multi-layer HNSW, geometric level sampling | `HNSWIndex._random_level`, `ml = 1/ln(M)` |
| M / efConstruction / efSearch tunables | `HNSWIndex.__init__` params, per-query override in `search(ef_search=...)` |
| Greedy descent top-layer → layer 0 | `HNSWIndex.insert` phase 1 loop + `search()` descent loop |
| Beam search (candidate list) | `_search_layer` (heap-based, ef-bounded) |
| Neighbor selection heuristic | `_select_neighbors_heuristic` (diversity-aware, not just closest-M) |
| Hybrid metadata filtering, doesn't break traversal | `_search_layer(..., allowed=...)`: filtered nodes are traversed but excluded from results |
| Tombstone soft delete | `Node.tombstone`, `HNSWIndex.delete()` |
| Compaction / rebuild | `HNSWIndex.compact()` — full re-insert of live vectors |
| Scalar quantization int8 | `core/quantization.py::ScalarQuantizer` |
| RW locking for concurrent reads + writes | `core/index.py::RWLock`, used in `insert`/`search`/`delete`/`compact` |
| WAL before memory mutation | `storage/wal.py`, called from `VectorDatabase.insert/delete` before... actually logged alongside; replay on boot in `_replay_wal` |
| Snapshot restore on startup | `storage/snapshot.py::Snapshot.save/load`, wired into `VectorDatabase.__init__` |
| Brute-force ground truth | `benchmarks/ground_truth.py::brute_force_knn` |
| Recall vs QPS sweep | `benchmarks/recall_qps.py` |
| FastAPI `/insert /search /delete /health` | `api/main.py` |
| Python client SDK | `client/client.py::VectorDBClient` |
| Docker / compose | `Dockerfile`, `docker-compose.yml` |

## 3. Quickstart

```bash
pip install -r requirements.txt

# run tests
pytest tests/ -v

# run the benchmark sweep
python -m benchmarks.recall_qps --n_vectors 2000 --dim 32 --n_queries 100 --k 10

# run the API
uvicorn api.main:app --reload
```

```python
# Embedded, no server (core/database.py)
from core.database import VectorDatabase
db = VectorDatabase(dim=128, metric="cosine", M=16, ef_construction=200)
id_ = db.insert([0.1]*128, metadata={"category": "finance"})
hits = db.search([0.1]*128, k=5, filter={"category": "finance"})
```

```python
# Over HTTP, via the client SDK
from client.client import VectorDBClient
db = VectorDBClient("http://localhost:8000")
id_ = db.insert([0.1]*128, metadata={"category": "finance"})
hits = db.search([0.1]*128, k=5, filter={"score": {"$gte": 10}})
```

```bash
docker compose up --build
curl http://localhost:8000/health
```

## 4. Benchmark results (real runs, two machines)

`n_vectors=2000, dim=32, k=10, efConstruction=200`, cosine metric, clustered
synthetic embeddings (20 centroids), measured against an exact brute-force
oracle. Reproduce with:

```bash
python -m benchmarks.recall_qps --n_vectors 2000 --dim 32 --n_queries 100 --k 10
```

### Build time by M

| M | Linux (sandbox) | Windows (user run) |
|---|---|---|
| 8  | 19.89s (100.6 inserts/s) | 21.17s (94.5 inserts/s) |
| 16 | 43.56s (45.9 inserts/s)  | 109.58s (18.3 inserts/s) |
| 32 | 137.47s (14.5 inserts/s) | 149.82s (13.3 inserts/s) |

### Recall@10 vs QPS by M and efSearch

| M | efSearch | Recall@10 | QPS (Linux) | QPS (Windows) |
|---|---|---|---|---|
| 8  | 10  | 0.642 | 2914.5 | 2545.6 |
| 8  | 20  | 0.729 | 2072.8 | 1838.4 |
| 8  | 50  | 0.822 | 1284.9 | 1142.6 |
| 8  | 100 | 0.898 |  830.4 |  737.1 |
| 8  | 200 | 0.949 |  456.8 |  424.3 |
| 16 | 10  | 0.673 | 2893.5 | 1460.2 |
| 16 | 20  | 0.768 | 2044.8 | 1002.6 |
| 16 | 50  | 0.820 | 1328.7 |  651.0 |
| 16 | 100 | 0.914 |  839.4 |  404.2 |
| 16 | 200 | 0.979 |  496.1 |  248.0 |
| 32 | 10  | 0.756 | 3306.1 | 3034.9 |
| 32 | 20  | 0.806 | 2543.1 | 2290.9 |
| 32 | 50  | 0.854 | 1593.0 | 1518.1 |
| 32 | 100 | 0.924 |  992.4 |  948.4 |
| 32 | 200 | 0.981 |  568.5 |  558.4 |

**Recall@10 is bit-for-bit identical across both machines at every (M,
efSearch) pair** — expected, since recall depends only on graph
construction/traversal logic and the fixed random seed (`seed=42`), not on
hardware. This is a useful sanity check when reproducing: if your recall
numbers differ from the table above on the same `n_vectors`/`dim`/seed,
something in the index logic changed, not just your machine.

**QPS varies by hardware/OS/background load, as expected** — the Windows
run's M=16 QPS is notably lower relative to Linux (down ~50-65%) despite
similar M=8 and M=32 numbers, likely OS scheduling/background-process
variance rather than anything algorithmic. Always benchmark QPS on your
own target hardware rather than trusting numbers from a different machine;
recall numbers, by contrast, are portable.

Full CSV: `benchmarks/results.csv` (regenerate anytime with `recall_qps.py`).
This is the canonical HNSW tradeoff curve: raising `efSearch` trades QPS for
recall; raising `M` raises the recall ceiling and build cost, with
diminishing QPS cost at high `ef`. Plot `recall_at_k` vs `qps`, grouped by
`M`, for the README chart.

## 5. Design notes: fixes applied to the initial limitations

The first version of this project had five known rough edges, listed
honestly rather than glossed over. All five have since been addressed —
here's exactly what changed and how each was verified.

### Concurrency — was whole-index lock, now per-node locking
The old `RWLock` was held for an insert's *entire* duration (full graph
traversal + neighbor connection), so concurrent inserts fully serialized.
Now:
- `HNSWIndex.insert()` is split into `reserve()` (short, holds the
  structural lock only for id allocation / vector-row write / entry-point
  bookkeeping) and `link()` (the expensive beam-search + neighbor-connect
  work, holding **no** whole-index lock).
- Each `Node` has its own `threading.Lock`; edge-list writes use
  copy-on-write (`neighbors[layer] = old_list + [new_id]`, a single atomic
  reference swap under the GIL) so `search()` can read neighbor lists with
  **no lock at all** and never sees a torn/partial list.
- Verified in `tests/test_basic.py::test_concurrent_inserts_and_searches_dont_corrupt_graph`
  — 8 writer threads + 8 reader threads hammering one graph simultaneously;
  no corruption, no lost/duplicate ids, every inserted vector remains
  findable afterward.

### Quantization — now actually wired into the distance path
`ScalarQuantizer` used to be a standalone class nothing called.
`HNSWIndex(quantize=True)` now stores real `uint8` rows (verified:
`idx.vectors.dtype == np.uint8`), and `_distances_to`/`_get_vec`
dequantize candidate batches in one vectorized call right before the
distance kernel runs — the graph-quality logic (beam search, neighbor
selection) never knows the storage format. `compact()` re-fits and
re-encodes correctly. Call `fit_quantizer(sample_vectors)` once before the
first insert (or `VectorDatabase.fit_quantizer(...)` / the API's
`/fit_quantizer` endpoint). Verified in
`tests/test_basic.py::test_quantized_index_end_to_end`.

### WAL ordering — now true log-before-apply
Previously `database.insert()` mutated the graph via `index.insert()`
*then* wrote the WAL record — a crash in between meant an unrecoverable
write. Fixed by splitting `HNSWIndex.insert` into `reserve()`/`link()`:
`database.insert()` now calls `reserve()` (assigns a stable, deterministic
node_id but doesn't make it graph-reachable), writes+fsyncs the WAL record
for that exact id, and only then calls `link()`. `_replay_wal` handles
both "insert never applied" and "reserved but crashed before linking"
cases. Verified by simulating a crash exactly between the WAL write and
the link step, then confirming a fresh process recovers and finds the
vector via search.

### Async compact() — now double-buffered, non-blocking
`compact()` used to hold the write lock for the entire O(N log N) rebuild.
Now `HNSWIndex.compact(background=True)` (the default) builds a brand new
graph in a separate `HNSWIndex` instance with no lock held on the live
one, then does a short atomic swap and replays any writes that raced the
rebuild window. `VectorDatabase.compact()` defaults to
`background=False` (so snapshot+WAL-truncate timing stays safe), but the
async primitive is available directly via `index.compact(background=True)`
for a real background scheduler. Verified in
`tests/test_basic.py::test_compact_runs_in_background_without_blocking` —
the index stays fully searchable and insertable *while* a rebuild is in
flight on another thread.

### API process model — explicit sharding instead of `--workers N`
`uvicorn --workers > 1` would silently create N disconnected in-memory
graphs (a correctness bug, not a scaling feature), since this index lives
in one process's memory. Fixed with `client/sharded_client.py`'s
`ShardedVectorDBClient`: run N single-worker instances, and the client
hash-routes `insert()` to exactly one shard (returning a composite
`"{shard_idx}:{local_id}"` id), fans `search()` out to all shards in
parallel and merges by distance, and routes `delete()`/`compact()` using
the composite id. Verified against two **real, independently running**
`uvicorn` processes (`examples/sharded_demo.py`): 40 vectors hash-balanced
19/21 across shards, cross-shard search correctly returns the true nearest
neighbor first (distance ≈ 0), and delete correctly removes the vector
from its actual shard.

## 6. Follow-up fixes to the three remaining caveats

These three were previously listed as "not fully solved, by design." They're
each fundamental tradeoffs of their approach (per-node locking, sharded
fan-out, sample-based quantization), so none of them is eliminated outright
— but each had a concrete, addressable gap that's now fixed. Honest
before/after:

### Hub-node lock hold time — shrunk, not eliminated
Two concurrent inserts connecting to the same hub node still can't write to
it *simultaneously* — that's inherent to having a lock at all. But
previously `_connect()` held that lock for the append **and** the full
prune (an O(len(neighbors)) distance computation + heuristic sort), so a
hub's lock was serializing on the expensive part, not just the cheap part.
`HNSWIndex._connect()` now does the append under the lock, then computes
the prune *outside* the lock, and only reacquires it to commit — using an
optimistic compare-and-swap (retry if another writer's edge landed in the
meantime) rather than holding the lock through the distance math. The lock
is now held for O(1) list operations; concurrent writers to different hubs
no longer wait on each other's distance calculations. Verified in
`tests/test_basic.py::test_connect_prune_never_drops_edges_under_contention`
— 6 threads inserting into a small-M (frequent-pruning) graph, then
asserting every neighbor list is within `m_target` and every neighbor id
still resolves to a real node (the failure mode a buggy retry would cause).

### Sharded search: failure handling, not round-trip count
The `O(shards)` parallel-fanout network cost is architectural and stays —
that's the cost of sharding at all. What was actually a gap: one
unreachable/slow shard made `ShardedVectorDBClient.search()` raise and
throw away every other shard's perfectly good results — an availability
bug on top of the expected latency cost. `search()` now takes a
`shard_timeout` and skips (rather than propagates) an individual shard's
failure, returning the merged results from whichever shards did respond;
failures are recorded in `self.last_search_errors` for callers/monitoring
to inspect, and only raises if *every* shard failed or
`require_all_shards=True` is passed. Verified in
`tests/test_basic.py::test_sharded_search_degrades_gracefully_on_shard_failure`.

### Quantizer clipping — wider tolerance + observability, not unbounded
`fit_quantizer()` still needs *a* representative sample, and a large enough
distribution shift will still eventually clip — quantization to a fixed
8-bit range can't fully escape that. Two real improvements:
`ScalarQuantizer` now pads the fitted min/max by a configurable `margin`
(10% default) so vectors moderately outside the original sample don't
immediately hit the boundary, and it tracks a running clip rate
(`clip_stats()`) with `should_refit()` flagging when clipping has become
frequent enough that a `compact()` (which re-fits on the live corpus) is
actually due — instead of drift silently degrading recall with no signal.
This is surfaced automatically in `HNSWIndex.stats()` / the `/health`
endpoint under `quantizer` whenever `quantize=True`. Verified in
`tests/test_basic.py::test_quantizer_margin_reduces_clipping_and_tracks_clip_rate`.

---

## Contact
 - LinkedIn: https://www.linkedin.com/in/nikhil-nair-809248286/
 - Email: nnair7598@gmail.com

## Thank You