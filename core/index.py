"""
core/index.py
HNSW (Hierarchical Navigable Small World) index.

Implements:
  - Multi-layer probabilistic graph construction (Malkov & Yashunin, 2016)
  - Greedy search with layer descent
  - Tunable M / efConstruction / efSearch
  - Tombstone soft-deletes + compaction/rebuild
  - Metadata (bitmap) filtered search that does not break traversal
  - Read-Write locking for concurrent readers / exclusive writers
  - Hooks for scalar-quantized distance (see core/quantization.py)
"""
from __future__ import annotations
import heapq
import math
import random
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .distance import Metric, batch_distance
from .node import Node
from .quantization import ScalarQuantizer


# --------------------------------------------------------------------------- #
# Structural Read-Write lock
# --------------------------------------------------------------------------- #
class RWLock:
    """Guards only the index's *structural* metadata: the vector-storage
    array (growth/writes to a fresh row), the `nodes` dict itself (adding a
    new id), `entry_point`, and `max_layer`. It does NOT guard per-node
    neighbor lists -- those are protected individually by `Node.lock` using
    copy-on-write list swaps (see core/node.py), and `search()` takes no
    lock at all.

    This is the fix for the old whole-index coarse lock: previously every
    insert held this lock for its *entire* duration (including graph
    traversal + neighbor selection for every layer), so concurrent inserts
    fully serialized. Now the lock is only held for the few-microsecond
    bookkeeping steps; the expensive beam-search + neighbor-connect work for
    two different target nodes can run in parallel, only serializing on the
    much smaller per-node locks if they happen to touch the same node.
    """

    def __init__(self):
        self._readers = 0
        self._read_ready = threading.Condition(threading.Lock())
        self._writer_active = False

    def acquire_read(self):
        with self._read_ready:
            while self._writer_active:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        self._read_ready.acquire()
        while self._readers > 0 or self._writer_active:
            self._read_ready.wait()
        self._writer_active = True

    def release_write(self):
        self._writer_active = False
        self._read_ready.notify_all()
        self._read_ready.release()

    class _ReadCtx:
        def __init__(self, lock): self.lock = lock
        def __enter__(self): self.lock.acquire_read()
        def __exit__(self, *a): self.lock.release_read()

    class _WriteCtx:
        def __init__(self, lock): self.lock = lock
        def __enter__(self): self.lock.acquire_write()
        def __exit__(self, *a): self.lock.release_write()

    def read(self): return RWLock._ReadCtx(self)
    def write(self): return RWLock._WriteCtx(self)


# --------------------------------------------------------------------------- #
# HNSW Index
# --------------------------------------------------------------------------- #
class HNSWIndex:
    def __init__(
        self,
        dim: int,
        metric: Metric | str = Metric.COSINE,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_elements: int = 100_000,
        seed: int = 42,
        quantize: bool = False,
    ):
        self.dim = dim
        self.metric = Metric(metric)
        self.M = M
        self.M0 = M * 2                 # layer-0 gets double the edges (standard HNSW heuristic)
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1.0 / math.log(M)     # level-generation normalization factor

        self._rng = random.Random(seed)
        self.lock = RWLock()

        # Contiguous vector storage: pre-allocated, grown geometrically.
        # quantize=True stores int8 (uint8) rows instead of float32 -- ~4x
        # less RAM. Distance math always happens in float32: candidate rows
        # are dequantized in a single vectorized batch right before the
        # distance kernel runs (see _distances_to), so HNSW's graph-quality
        # logic (greedy descent, beam search, neighbor selection) is
        # completely unaware of the storage format.
        self.quantize = quantize
        self.quantizer: Optional[ScalarQuantizer] = ScalarQuantizer(dim) if quantize else None
        self._capacity = max_elements
        store_dtype = np.uint8 if quantize else np.float32
        self.vectors = np.zeros((self._capacity, dim), dtype=store_dtype)
        self._size = 0                  # number of rows actually used

        self.nodes: Dict[int, Node] = {}
        self.entry_point: Optional[int] = None
        self.max_layer: int = -1
        self._free_ids: List[int] = []  # ids freed by compaction, reused on insert
        self._next_id = 0

    def fit_quantizer(self, sample_vectors: np.ndarray) -> None:
        """Must be called once, before the first insert, when quantize=True.
        Fits per-dimension min/max on a representative sample so int8
        encoding has sane range (see core/quantization.py). Re-fitting
        later (e.g. during compact()) is possible but not automatic, since
        it would require re-encoding every stored vector."""
        if not self.quantize:
            raise RuntimeError("fit_quantizer() only applies when quantize=True")
        self.quantizer.fit(np.asarray(sample_vectors, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Vector storage helpers
    # ------------------------------------------------------------------ #
    def _grow_if_needed(self):
        if self._size >= self._capacity:
            new_cap = self._capacity * 2
            grown = np.zeros((new_cap, self.dim), dtype=self.vectors.dtype)
            grown[: self._capacity] = self.vectors
            self.vectors = grown
            self._capacity = new_cap

    def _random_level(self) -> int:
        # Standard HNSW geometric distribution for layer assignment.
        return int(-math.log(self._rng.random() + 1e-12) * self.ml)

    def _get_vec(self, node_id: int) -> np.ndarray:
        """Always returns a float32 vector, dequantizing on the fly if this
        index stores int8 rows. Single-row dequantize is cheap; the hot
        path (batch search) uses _distances_to's batched dequantize instead."""
        row = self.vectors[node_id]
        if self.quantize:
            return self.quantizer.decode(row)
        return row

    def _distances_to(self, query: np.ndarray, candidate_ids: List[int]) -> np.ndarray:
        if not candidate_ids:
            return np.array([])
        batch = self.vectors[candidate_ids]
        if self.quantize:
            # One vectorized dequantize for the whole candidate batch, then
            # the normal float32 distance kernel -- HNSW's traversal logic
            # never has to know vectors were stored as int8.
            batch = self.quantizer.decode_batch(batch)
        return batch_distance(query, batch, self.metric)

    # ------------------------------------------------------------------ #
    # Insert -- split into reserve() + link() so a caller (VectorDatabase)
    # can write a durable WAL record for the *exact* node_id in between the
    # two steps, giving true log-before-apply WAL semantics instead of
    # logging after the graph mutation is already visible to searches.
    # ------------------------------------------------------------------ #
    def reserve(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Allocate a node id and write its vector row + bare Node object.
        The node exists in `self.nodes` but has no edges yet, so it is not
        yet reachable via graph traversal (unless it's the very first node,
        in which case it immediately becomes the entry point). Cheap and
        fully structural -- this is the only part that needs the WAL record
        to exist *before* it runs, per the durability contract "if the WAL
        record exists, recovery can reconstruct the operation": the id is
        stable and deterministic the instant this returns, before any
        graph work happens."""
        vector = np.asarray(vector, dtype=np.float32)
        assert vector.shape == (self.dim,), f"expected dim {self.dim}, got {vector.shape}"
        if self.quantize:
            if not self.quantizer.fitted:
                raise RuntimeError(
                    "quantize=True but fit_quantizer(sample_vectors) was never called; "
                    "fit it on a representative sample before the first insert."
                )
            stored_row = self.quantizer.encode(vector)
        else:
            stored_row = vector

        with self.lock.write():
            node_id = self._free_ids.pop() if self._free_ids else self._next_id
            if node_id == self._next_id:
                self._next_id += 1
            self._grow_if_needed()
            self.vectors[node_id] = stored_row
            self._size = max(self._size, node_id + 1)

            level = self._random_level()
            self.nodes[node_id] = Node(id=node_id, level=level, metadata=metadata or {})

            if self.entry_point is None:
                self.entry_point = node_id
                self.max_layer = level

        return node_id

    def link(self, node_id: int) -> None:
        """Wire a reserved node into the graph: greedy descent + beam
        search + neighbor connection at every layer up to its assigned
        level. No-op if this node happens to be the very first node in
        the index (already made the entry point by reserve())."""
        node = self.nodes[node_id]
        vector = self._get_vec(node_id)  # dequantized float32 if quantize=True
        level = node.level

        with self.lock.write():
            ep, top_layer = self.entry_point, self.max_layer
        if ep == node_id:
            return  # first node in the index; nothing to link to

        cur_dist = self._distances_to(vector, [ep])[0]
        for layer in range(top_layer, level, -1):
            ep, cur_dist = self._greedy_step(vector, ep, cur_dist, layer)

        for layer in range(min(level, top_layer), -1, -1):
            candidates = self._search_layer(vector, ep, self.ef_construction, layer)
            m_target = self.M0 if layer == 0 else self.M
            selected = self._select_neighbors_heuristic(vector, candidates, m_target)

            node.set_neighbors(layer, [c for c, _ in selected])
            for cand_id, _ in selected:
                self._connect(cand_id, node_id, layer)

            if candidates:
                ep = candidates[0][0]

        if level > top_layer:
            with self.lock.write():
                if level > self.max_layer:  # re-check: another writer may have raced us
                    self.max_layer = level
                    self.entry_point = node_id

    def insert(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Convenience wrapper: reserve + link in one call. Used directly by
        tests/benchmarks/compaction. VectorDatabase calls reserve()/link()
        separately so it can log to the WAL in between."""
        node_id = self.reserve(vector, metadata)
        self.link(node_id)
        return node_id

    def _greedy_step(self, query, ep, ep_dist, layer) -> Tuple[int, float]:
        improved = True
        while improved:
            improved = False
            neighbors = [n for n in self.nodes[ep].get_neighbors(layer) if not self.nodes[n].tombstone]
            if not neighbors:
                break
            dists = self._distances_to(query, neighbors)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < ep_dist:
                ep, ep_dist = neighbors[min_idx], float(dists[min_idx])
                improved = True
        return ep, ep_dist

    def _search_layer(self, query, entry_id, ef, layer, allowed: Optional[Callable[[int], bool]] = None):
        """Beam search on a single layer. Returns list[(id, dist)] sorted ascending,
        length <= ef. `allowed` is an optional predicate for metadata-filtered
        search: nodes failing it are still *traversed through* (so the graph
        stays connected) but never enter the result candidate set."""
        visited = {entry_id}
        ep_dist = float(self._distances_to(query, [entry_id])[0])

        candidates = [(ep_dist, entry_id)]  # min-heap
        heapq.heapify(candidates)

        results: List[Tuple[float, int]] = []
        if not self.nodes[entry_id].tombstone and (allowed is None or allowed(entry_id)):
            results.append((-ep_dist, entry_id))  # max-heap via negation

        while candidates:
            cur_dist, cur_id = heapq.heappop(candidates)
            if results and cur_dist > -results[0][0]:
                break  # no closer candidates possible

            neighbors = [n for n in self.nodes[cur_id].get_neighbors(layer) if n not in visited]
            if not neighbors:
                continue
            visited.update(neighbors)
            dists = self._distances_to(query, neighbors)

            for nb_id, nb_dist in zip(neighbors, dists):
                nb_dist = float(nb_dist)
                if len(results) < ef or nb_dist < -results[0][0]:
                    heapq.heappush(candidates, (nb_dist, nb_id))
                    node_ok = not self.nodes[nb_id].tombstone and (allowed is None or allowed(nb_id))
                    if node_ok:
                        heapq.heappush(results, (-nb_dist, nb_id))
                        if len(results) > ef:
                            heapq.heappop(results)

        results.sort(key=lambda x: -x[0])
        return [(nid, -d) for d, nid in results]

    def _select_neighbors_heuristic(self, query, candidates: List[Tuple[int, float]], m: int):
        """Simple heuristic neighbor selection: closest-first with a diversity
        check (a candidate is skipped if it's closer to an already-selected
        neighbor than to the query -- avoids clustering all edges in one
        direction, per the original HNSW paper's heuristic selection)."""
        candidates = sorted(candidates, key=lambda x: x[1])
        selected: List[Tuple[int, float]] = []
        for cand_id, cand_dist in candidates:
            if len(selected) >= m:
                break
            competitive = True
            if selected:
                sel_ids = [s[0] for s in selected]
                d_to_selected = self._distances_to(self._get_vec(cand_id), sel_ids)
                if np.min(d_to_selected) < cand_dist:
                    competitive = False
            if competitive:
                selected.append((cand_id, cand_dist))
        # backfill if the diversity check was too aggressive
        if len(selected) < m:
            chosen_ids = {c for c, _ in selected}
            for cand_id, cand_dist in candidates:
                if len(selected) >= m:
                    break
                if cand_id not in chosen_ids:
                    selected.append((cand_id, cand_dist))
        return selected

    def _connect(self, a: int, b: int, layer: int):
        """Add a<->b edge and prune a's neighbor list back down to m_target
        if it overflowed. The add + read + prune sequence is wrapped in
        node `a`'s own lock so a concurrent _connect(a, c, layer) from
        another insert can't interleave and prune away b's freshly-added
        edge based on a stale neighbor list (a lost-update race)."""
        node_a = self.nodes[a]
        with node_a.lock:
            current = node_a.neighbors.get(layer, [])
            if b not in current:
                node_a.neighbors[layer] = current + [b]

            m_target = self.M0 if layer == 0 else self.M
            neighbors = node_a.neighbors.get(layer, [])
            if len(neighbors) > m_target:
                dists = self._distances_to(self._get_vec(a), neighbors)
                ranked = sorted(zip(neighbors, dists), key=lambda x: x[1])
                pruned = self._select_neighbors_heuristic(self._get_vec(a), ranked, m_target)
                node_a.neighbors[layer] = [c for c, _ in pruned]

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Return top-k (id, distance, metadata) approximate nearest neighbors.
        `filter_fn(metadata) -> bool` implements hybrid/metadata filtering:
        the graph traversal still visits filtered-out nodes as stepping
        stones, but they never appear in the returned result set."""
        query = np.asarray(query, dtype=np.float32)
        ef = ef_search or self.ef_search
        ef = max(ef, k)

        # Deliberately lock-free: reads a consistent (entry_point, max_layer)
        # snapshot, then traverses. Node.neighbors reads are always a full,
        # up-to-date-or-previous list reference (copy-on-write in Node), so
        # a concurrent insert can never hand this search a torn/partial
        # neighbor list -- worst case it searches a graph that's a few
        # edges "behind" the very latest writes, which is fine for an ANN
        # index (approximate by definition) and avoids blocking readers
        # behind writers entirely.
        ep = self.entry_point
        top_layer = self.max_layer
        if ep is None:
            return []

        allowed = None
        if filter_fn is not None:
            allowed = lambda nid: filter_fn(self.nodes[nid].metadata)  # noqa: E731

        ep_dist = float(self._distances_to(query, [ep])[0])
        for layer in range(top_layer, 0, -1):
            ep, ep_dist = self._greedy_step(query, ep, ep_dist, layer)

        candidates = self._search_layer(query, ep, ef, 0, allowed=allowed)
        top = candidates[:k]
        return [(nid, dist, self.nodes[nid].metadata) for nid, dist in top]

    # ------------------------------------------------------------------ #
    # Delete (tombstone) + compaction
    # ------------------------------------------------------------------ #
    def delete(self, node_id: int) -> bool:
        with self.lock.write():
            node = self.nodes.get(node_id)
            if node is None or node.tombstone:
                return False
            node.tombstone = True
            if node_id == self.entry_point:
                self._reassign_entry_point()
            return True

    def _reassign_entry_point(self):
        for nid, n in self.nodes.items():
            if not n.tombstone:
                self.entry_point = nid
                self.max_layer = n.level
                return
        self.entry_point = None
        self.max_layer = -1

    def compact(self, background: bool = True):
        """Purge tombstoned nodes and re-index remaining vectors.

        Double-buffered rebuild: the new graph is built in a *separate*
        HNSWIndex instance with no lock held on `self`, so normal
        insert/search/delete traffic on the live index is never blocked
        for the O(N log N) rebuild duration -- only for two short
        snapshot/swap critical sections. Any writes that land on `self`
        during the rebuild window are captured by a diff after the swap
        and replayed on top of the new graph.

        `background=True` (default) runs this on a daemon thread and
        returns immediately, matching the "periodic background compaction"
        requirement. `background=False` runs synchronously and blocks
        until done (useful in tests/scripts where you want the result
        before proceeding).
        """
        if background:
            t = threading.Thread(target=self._compact_and_swap, daemon=True)
            t.start()
            return t
        self._compact_and_swap()
        return None

    def _compact_and_swap(self):
        with self.lock.write():
            live_ids = [nid for nid, n in self.nodes.items() if not n.tombstone]
            # _get_vec dequantizes per-row if this index is quantized, so
            # `live_vectors` is always plain float32 regardless of storage mode.
            live_vectors = (
                np.stack([self._get_vec(nid) for nid in live_ids])
                if live_ids else np.zeros((0, self.dim), np.float32)
            )
            live_meta = [self.nodes[nid].metadata for nid in live_ids]
            snapshot_ids = set(live_ids)

        new_index = HNSWIndex(
            dim=self.dim, metric=self.metric, M=self.M,
            ef_construction=self.ef_construction, ef_search=self.ef_search,
            max_elements=max(len(live_ids) * 2, 1000),
            seed=self._rng.randrange(1, 2**31),
            quantize=self.quantize,
        )
        if self.quantize and live_vectors.shape[0] > 0:
            # Re-fit on the live corpus (min/max may have drifted since the
            # original fit) rather than reusing the old quantizer's range.
            new_index.fit_quantizer(live_vectors)
        for vec, meta in zip(live_vectors, live_meta):
            new_index.insert(vec, meta)

        # Atomic-ish swap: adopt the freshly built graph's internals.
        with self.lock.write():
            currently_live = {nid for nid, n in self.nodes.items() if not n.tombstone}
            missed_ids = currently_live - snapshot_ids  # writes that raced the rebuild
            # dequantize with the OLD quantizer (self.vectors/self.quantizer
            # haven't been swapped yet at this point) so `missed` is plain
            # float32, ready to be re-encoded with the NEW quantizer below.
            missed = [(self._get_vec(nid).copy(), self.nodes[nid].metadata) for nid in missed_ids]

            self.vectors = new_index.vectors
            self._capacity = new_index._capacity
            self.nodes = new_index.nodes
            self.entry_point = new_index.entry_point
            self.max_layer = new_index.max_layer
            self._size = new_index._size
            self._next_id = new_index._next_id
            self._free_ids = new_index._free_ids
            if self.quantize:
                self.quantizer = new_index.quantizer  # possibly re-fit range

        # Replay anything inserted during the rebuild window onto the new graph.
        for vec, meta in missed:
            self.insert(vec, meta)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def __len__(self):
        return sum(1 for n in self.nodes.values() if not n.tombstone)

    def stats(self) -> Dict[str, Any]:
        return {
            "total_nodes": len(self.nodes),
            "live_nodes": len(self),
            "tombstoned": len(self.nodes) - len(self),
            "max_layer": self.max_layer,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "dim": self.dim,
            "metric": self.metric.value,
        }
