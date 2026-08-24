"""
storage/snapshot.py
Point-in-time binary snapshot of an HNSWIndex, so the database can restore
full graph state on startup without replaying the entire WAL / rebuilding
the graph from scratch. Pattern used in production systems:

    on startup:
        index = Snapshot.load(path) if exists else HNSWIndex(...)
        wal.replay_since(snapshot_ts) -> apply any newer ops
    periodically (e.g. every N writes or every T seconds):
        Snapshot.save(index, path)
        wal.truncate()
"""
from __future__ import annotations
import pickle
import shutil
import tempfile
import time
from pathlib import Path

from core.index import HNSWIndex


class Snapshot:
    @staticmethod
    def save(index: HNSWIndex, path: str) -> None:
        """Atomic write: dump to a temp file, then rename over the target so
        a crash mid-write never leaves a corrupt snapshot on disk."""
        with index.lock.read():
            state = {
                "dim": index.dim,
                "metric": index.metric.value,
                "M": index.M,
                "ef_construction": index.ef_construction,
                "ef_search": index.ef_search,
                "vectors": index.vectors[: index._size].copy(),
                "size": index._size,
                "nodes": index.nodes,
                "entry_point": index.entry_point,
                "max_layer": index.max_layer,
                "next_id": index._next_id,
                "free_ids": index._free_ids,
                "saved_at": time.time(),
            }
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=str(Path(path).parent) or ".")
        try:
            pickle.dump(state, tmp)
            tmp.flush()
            tmp.close()
            shutil.move(tmp.name, path)
        finally:
            if Path(tmp.name).exists():
                Path(tmp.name).unlink()

    @staticmethod
    def load(path: str) -> HNSWIndex:
        with open(path, "rb") as f:
            state = pickle.load(f)

        index = HNSWIndex(
            dim=state["dim"],
            metric=state["metric"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            ef_search=state["ef_search"],
            max_elements=max(state["size"] * 2, 1000),
        )
        index.vectors[: state["size"]] = state["vectors"]
        index._size = state["size"]
        index.nodes = state["nodes"]
        index.entry_point = state["entry_point"]
        index.max_layer = state["max_layer"]
        index._next_id = state["next_id"]
        index._free_ids = state["free_ids"]
        return index

    @staticmethod
    def exists(path: str) -> bool:
        return Path(path).exists()
