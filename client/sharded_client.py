"""
client/sharded_client.py
Fix for issue #4 in the README limitations: a single VectorDatabase's HNSW
graph lives in ONE process's memory, so `uvicorn --workers > 1` silently
creates N *independent, disconnected* graphs (each worker only sees the
writes routed to it by the OS, and reads only see its own shard) --
that's a correctness bug, not a scaling feature.

The fix is explicit application-level sharding: run N single-worker
VectorDB instances (N separate `docker compose` services / processes,
each with `--workers 1` and its own VDB_DATA_DIR), and route to them from
the client:

  - insert(): hashes a routing key to pick exactly one shard, deterministically.
  - search(): fans out to ALL shards in parallel, merges results by distance,
              returns the global top-k. Correct because HNSW distance is
              comparable across independently-built graphs on the same metric.
  - delete(): needs to know which shard an id lives on, so insert() returns
              a composite id "{shard_index}:{local_node_id}" instead of a
              bare int, and delete()/anything id-based parses it back out.

This trades a bit of id ergonomics (composite ids) for horizontal scale-out
that's actually correct, instead of `--workers N` which is silently wrong.
"""
from __future__ import annotations
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .client import VectorDBClient


def _composite_id(shard_index: int, local_id: int) -> str:
    return f"{shard_index}:{local_id}"


def _parse_composite_id(composite_id: str) -> tuple[int, int]:
    shard_str, local_str = composite_id.split(":", 1)
    return int(shard_str), int(local_str)


class ShardedVectorDBClient:
    def __init__(self, base_urls: List[str], timeout: float = 10.0, max_workers: Optional[int] = None):
        """base_urls: one URL per independently-running VectorDB API
        instance (e.g. N docker-compose services on different ports),
        each started with the SAME dim/metric config -- sharding assumes
        all shards are otherwise-identical index configurations."""
        if not base_urls:
            raise ValueError("ShardedVectorDBClient needs at least one base_url")
        self.shards = [VectorDBClient(url, timeout=timeout) for url in base_urls]
        self.n_shards = len(self.shards)
        self._pool = ThreadPoolExecutor(max_workers=max_workers or self.n_shards)

    def _shard_for_key(self, key: str) -> int:
        # Deterministic hash -> shard index (consistent for a given key, so
        # repeated inserts with the same routing key always land on the
        # same shard -- useful for e.g. co-locating a user's vectors).
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.n_shards

    # ------------------------------------------------------------------ #
    def insert(
        self,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        routing_key: Optional[str] = None,
    ) -> str:
        """Returns a COMPOSITE id ("shard_idx:local_id"), not a bare int --
        callers must pass this exact string to delete(), not the local id.
        `routing_key`: if omitted, a random/round-robin-ish key derived
        from the vector's own bytes is used (fine for even distribution);
        pass an explicit key (e.g. a user id) to co-locate related vectors
        on the same shard."""
        key = routing_key if routing_key is not None else repr(vector)
        shard_idx = self._shard_for_key(key)
        local_id = self.shards[shard_idx].insert(vector, metadata)
        return _composite_id(shard_idx, local_id)

    def insert_batch(
        self,
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        routing_keys: Optional[List[str]] = None,
    ) -> List[str]:
        metadatas = metadatas or [{}] * len(vectors)
        routing_keys = routing_keys or [None] * len(vectors)
        return [
            self.insert(v, m, k)
            for v, m, k in zip(vectors, metadatas, routing_keys)
        ]

    def search(
        self,
        vector: List[float],
        k: int = 10,
        ef_search: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Fan out to every shard in parallel (each shard independently
        returns its own local top-k), then merge by distance and take the
        global top-k. Correctness note: this assumes all shards share the
        same metric/embedding space, so distances are directly comparable
        -- true as long as every shard was configured identically."""
        futures = {
            self._pool.submit(shard.search, vector, k, ef_search, filter): idx
            for idx, shard in enumerate(self.shards)
        }
        merged: List[Dict[str, Any]] = []
        for future in as_completed(futures):
            shard_idx = futures[future]
            hits = future.result()
            for hit in hits:
                merged.append({
                    "id": _composite_id(shard_idx, hit["id"]),
                    "distance": hit["distance"],
                    "metadata": hit["metadata"],
                })
        merged.sort(key=lambda h: h["distance"])
        return merged[:k]

    def delete(self, composite_id: str) -> bool:
        shard_idx, local_id = _parse_composite_id(composite_id)
        return self.shards[shard_idx].delete(local_id)

    def compact(self) -> List[Dict[str, Any]]:
        futures = [self._pool.submit(shard.compact) for shard in self.shards]
        return [f.result() for f in futures]

    def health(self) -> Dict[str, Any]:
        """Aggregate health/stats across all shards, plus per-shard detail
        so an operator can spot an unbalanced or unhealthy shard."""
        futures = {self._pool.submit(shard.health): idx for idx, shard in enumerate(self.shards)}
        per_shard = [None] * self.n_shards
        for future, idx in futures.items():
            try:
                per_shard[idx] = future.result()
            except Exception as e:  # noqa: BLE001
                per_shard[idx] = {"status": "unreachable", "error": str(e)}

        total_live = sum(
            s["stats"]["live_nodes"] for s in per_shard if s.get("status") == "ok"
        )
        return {
            "status": "ok" if all(s.get("status") == "ok" for s in per_shard) else "degraded",
            "n_shards": self.n_shards,
            "total_live_nodes": total_live,
            "shards": per_shard,
        }
