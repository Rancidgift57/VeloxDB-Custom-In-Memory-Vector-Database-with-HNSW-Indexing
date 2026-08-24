"""
examples/sharded_demo.py
Reproduces the sharding fix end-to-end against REAL, independent server
processes (not mocks). Run:

    # terminal 1
    VDB_DIM=8 VDB_DATA_DIR=/tmp/shard0 uvicorn api.main:app --port 8001

    # terminal 2
    VDB_DIM=8 VDB_DATA_DIR=/tmp/shard1 uvicorn api.main:app --port 8002

    # terminal 3
    python -m examples.sharded_demo

Why this exists: `uvicorn --workers N` would silently give you N
*disconnected* HNSW graphs sharing one process's routing but not one
process's memory -- reads/writes get randomly split across workers with
no coordination, which is a correctness bug. Running N single-worker
instances and routing explicitly from the client (this script) is the
correct way to scale this in-memory index horizontally.
"""
from __future__ import annotations
import numpy as np

from client.sharded_client import ShardedVectorDBClient


def main():
    db = ShardedVectorDBClient([
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
    ])

    rng = np.random.default_rng(0)
    vectors = rng.random((40, 8)).astype(np.float32)

    ids = [db.insert(v.tolist(), metadata={"i": int(i)}) for i, v in enumerate(vectors)]
    print(f"inserted {len(ids)} vectors across {db.n_shards} shards")

    shard_counts: dict[str, int] = {}
    for cid in ids:
        shard_idx = cid.split(":")[0]
        shard_counts[shard_idx] = shard_counts.get(shard_idx, 0) + 1
    print("distribution across shards:", shard_counts)

    health = db.health()
    print("aggregate health:", health["status"], "| total_live_nodes:", health["total_live_nodes"])

    query = vectors[3].tolist()
    hits = db.search(query, k=5)
    print("top-5 global search results (merged across shards):")
    for h in hits:
        print(f"  id={h['id']:<8} distance={h['distance']:.6f} metadata={h['metadata']}")

    target = ids[10]
    print(f"deleting {target} ...", "ok" if db.delete(target) else "failed")


if __name__ == "__main__":
    main()
