"""
storage/wal.py
Write-Ahead Log: every mutating operation (insert/delete) is appended to an
append-only log file BEFORE being applied to the in-memory index. On
startup, replaying the WAL from the last snapshot reconstructs any writes
that happened after that snapshot but before a crash.

Format: one JSON object per line (JSONL) -- simple, human-inspectable,
and append-friendly (no need to rewrite the whole file on each write).
"""
from __future__ import annotations
import json
import os
import threading
import time
from typing import Any, Dict, Iterator, Optional

import numpy as np


class WriteAheadLog:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)  # line-buffered

    def log_insert(self, node_id: int, vector: np.ndarray, metadata: Optional[Dict[str, Any]]):
        record = {
            "op": "insert",
            "ts": time.time(),
            "id": node_id,
            "vector": vector.astype(np.float32).tolist(),
            "metadata": metadata or {},
        }
        self._write(record)

    def log_delete(self, node_id: int):
        self._write({"op": "delete", "ts": time.time(), "id": node_id})

    def log_compact(self):
        self._write({"op": "compact", "ts": time.time()})

    def _write(self, record: Dict[str, Any]):
        with self._lock:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())  # durability: survive OS crash, not just process crash

    def replay(self) -> Iterator[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def truncate(self):
        """Called after a successful snapshot -- the WAL only needs to hold
        operations that happened *since* that snapshot."""
        with self._lock:
            self._fh.close()
            open(self.path, "w").close()
            self._fh = open(self.path, "a", buffering=1)

    def close(self):
        with self._lock:
            self._fh.close()
