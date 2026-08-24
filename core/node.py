"""
core/node.py
Node representation for the HNSW graph.

Design notes:
- Vectors themselves are NOT stored on the node; they live in a contiguous
  numpy matrix owned by the Index (see core/index.py). The node only stores
  the integer row-id into that matrix. This keeps distance computation
  vectorizable (batch gather + BLAS) instead of scattered python objects.
- `neighbors` is a dict: layer -> list[int] (neighbor ids at that layer).
- `tombstone` implements soft-delete: the node stays in the graph (so paths
  through it remain intact) but is skipped as a search *result* and is
  excluded from further outgoing edge selection during future inserts.
"""
from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Node:
    id: int                                  # row index into the vector matrix
    level: int                                # highest layer this node appears on
    metadata: Dict[str, Any] = field(default_factory=dict)
    neighbors: Dict[int, List[int]] = field(default_factory=dict)  # layer -> [ids]
    tombstone: bool = False
    # Fine-grained per-node lock guarding this node's own neighbor lists.
    # Not part of the dataclass fields (no sane default/repr for a Lock),
    # attached in __post_init__ instead.
    lock: threading.Lock = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.lock = threading.Lock()

    # threading.Lock objects aren't picklable, so exclude it from snapshot
    # serialization (storage/snapshot.py pickles the whole `nodes` dict) and
    # simply recreate a fresh lock on load -- the lock only protects
    # in-process concurrent writers, it carries no state that needs to
    # survive a restart.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def ensure_layer(self, layer: int) -> None:
        if layer not in self.neighbors:
            self.neighbors[layer] = []

    def add_neighbor(self, layer: int, neighbor_id: int) -> None:
        """Copy-on-write append: build a *new* list and swap the dict entry
        in one reference assignment (atomic under the GIL), rather than
        mutating the existing list in place. This lets concurrent readers
        (HNSWIndex.search, which does NOT take any lock) safely read
        `neighbors[layer]` at any time without ever seeing a torn/partial
        list -- they either see the old list or the new one, never a
        half-appended one. `self.lock` only serializes concurrent *writers*
        racing to update this same node."""
        with self.lock:
            current = self.neighbors.get(layer, [])
            if neighbor_id not in current:
                self.neighbors[layer] = current + [neighbor_id]

    def set_neighbors(self, layer: int, neighbor_ids: List[int]) -> None:
        with self.lock:
            self.neighbors[layer] = list(neighbor_ids)

    def get_neighbors(self, layer: int) -> List[int]:
        return self.neighbors.get(layer, [])
