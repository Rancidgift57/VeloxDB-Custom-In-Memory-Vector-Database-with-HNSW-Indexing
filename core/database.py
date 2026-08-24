"""
core/database.py
The facade the API and client SDK actually talk to. Wires together:
  HNSWIndex (in-memory graph)  +  WriteAheadLog (durability)  +  Snapshot (fast restart)
"""
from __future__ import annotations
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .index import HNSWIndex
from .distance import Metric
from filtering.metadata_filter import compile_filter
from storage.wal import WriteAheadLog
from storage.snapshot import Snapshot


class VectorDatabase:
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        data_dir: str = "./data",
        snapshot_every_n_writes: int = 1000,
        quantize: bool = False,
    ):
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir
        self.snapshot_path = os.path.join(data_dir, "snapshot.pkl")
        self.wal_path = os.path.join(data_dir, "wal.log")
        self.snapshot_every_n_writes = snapshot_every_n_writes
        self._writes_since_snapshot = 0
        self._snapshot_lock = threading.Lock()

        if Snapshot.exists(self.snapshot_path):
            self.index = Snapshot.load(self.snapshot_path)
        else:
            self.index = HNSWIndex(
                dim=dim, metric=metric, M=M,
                ef_construction=ef_construction, ef_search=ef_search,
                quantize=quantize,
            )

        self.wal = WriteAheadLog(self.wal_path)
        self._replay_wal()

    def fit_quantizer(self, sample_vectors) -> None:
        """Must be called once before the first insert if the database was
        constructed with quantize=True. See HNSWIndex.fit_quantizer."""
        self.index.fit_quantizer(np.asarray(sample_vectors, dtype=np.float32))

    # ------------------------------------------------------------------ #
    def _replay_wal(self):
        """Recovery path: re-apply any operations logged after the last
        snapshot. Safe to call on every startup -- inserts are idempotent
        per node_id and deletes are idempotent (tombstone is a no-op if
        already set). Because the log-before-apply ordering below
        guarantees a WAL record exists before the node is linked into the
        graph, a crash between reserve() and link() just means we replay
        an insert whose node_id was already reserved -- reserve() again
        would double-allocate, so replay always goes through the same
        `_apply_insert` path that reserve+link does live."""
        for record in self.wal.replay():
            if record["op"] == "insert":
                nid = record["id"]
                if nid not in self.index.nodes:
                    vec = np.array(record["vector"], dtype=np.float32)
                    self._apply_insert(vec, record["metadata"], node_id=nid)
                elif not self.index.nodes[nid].neighbors and self.index.entry_point != nid:
                    # node was reserved (row written) but never linked before
                    # the crash -- finish linking it now.
                    self.index.link(nid)
            elif record["op"] == "delete":
                self.index.delete(record["id"])

    def _apply_insert(self, vector, metadata, node_id: Optional[int] = None) -> int:
        # WAL replay path: re-create the node at the SAME id it originally
        # had. reserve() doesn't support a forced id, so during replay we
        # bypass it and write the row/Node directly at `node_id`, then link.
        if node_id is None:
            return self.index.insert(vector, metadata)
        self.index._grow_if_needed()
        self.index.vectors[node_id] = vector
        self.index._size = max(self.index._size, node_id + 1)
        self.index._next_id = max(self.index._next_id, node_id + 1)
        from core.node import Node
        level = self.index._random_level()
        self.index.nodes[node_id] = Node(id=node_id, level=level, metadata=metadata or {})
        if self.index.entry_point is None:
            self.index.entry_point = node_id
            self.index.max_layer = level
        else:
            self.index.link(node_id)
        return node_id

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def insert(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        """True log-before-apply: reserve a stable node_id (cheap, purely
        structural -- the node isn't reachable via search yet), write the
        WAL record for that exact id and vector, fsync it to disk, and
        ONLY THEN link the node into the graph. If the process crashes
        after reserve() but before/during link(), the WAL record is
        already durable on disk and `_replay_wal` finishes the link (or
        redoes the whole insert) on the next startup -- no write is ever
        acknowledged without first being durable."""
        vec = np.array(vector, dtype=np.float32)
        node_id = self.index.reserve(vec, metadata)
        self.wal.log_insert(node_id, vec, metadata)   # durable BEFORE linking
        self.index.link(node_id)
        self._maybe_snapshot()
        return node_id

    def insert_batch(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        metadatas = metadatas or [{}] * len(vectors)
        return [self.insert(v, m) for v, m in zip(vectors, metadatas)]

    def search(
        self,
        vector: List[float],
        k: int = 10,
        ef_search: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        query = np.array(vector, dtype=np.float32)
        filter_fn = compile_filter(filter) if filter else None
        results = self.index.search(query, k=k, ef_search=ef_search, filter_fn=filter_fn)
        return [{"id": nid, "distance": dist, "metadata": meta} for nid, dist, meta in results]

    def delete(self, node_id: int) -> bool:
        ok = self.index.delete(node_id)
        if ok:
            self.wal.log_delete(node_id)
            self._maybe_snapshot()
        return ok

    def compact(self, background: bool = False):
        """By default runs synchronously so `compact()` returning means the
        rebuild is done and it's safe to snapshot + truncate the WAL right
        after (their timing must not race a still-running rebuild). Pass
        `background=True` to kick off `HNSWIndex.compact()`'s double-
        buffered async rebuild instead -- in that mode the caller owns
        calling `snapshot()`/`_force_snapshot()` once the returned thread
        joins, since this method returns immediately without waiting."""
        thread = self.index.compact(background=background)
        if background:
            return thread  # caller decides when to snapshot
        self.wal.log_compact()
        self._force_snapshot()
        return None

    def stats(self) -> Dict[str, Any]:
        return self.index.stats()

    # ------------------------------------------------------------------ #
    def _maybe_snapshot(self):
        self._writes_since_snapshot += 1
        if self._writes_since_snapshot >= self.snapshot_every_n_writes:
            self._force_snapshot()

    def _force_snapshot(self):
        with self._snapshot_lock:
            Snapshot.save(self.index, self.snapshot_path)
            self.wal.truncate()
            self._writes_since_snapshot = 0

    def close(self):
        self._force_snapshot()
        self.wal.close()
