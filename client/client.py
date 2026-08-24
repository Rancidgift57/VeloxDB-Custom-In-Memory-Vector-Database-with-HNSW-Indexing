"""
client/client.py
Lightweight Python client SDK for the VectorDB API.

Usage:
    from client.client import VectorDBClient
    db = VectorDBClient("http://localhost:8000")
    id_ = db.insert([0.1, 0.2, ...], metadata={"category": "finance"})
    hits = db.search([0.1, 0.2, ...], k=5, filter={"category": "finance"})
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import requests


class VectorDBClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = self._session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def insert(self, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> int:
        out = self._post("/insert", {"vector": vector, "metadata": metadata or {}})
        return out["id"]

    def insert_batch(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        out = self._post("/insert_batch", {"vectors": vectors, "metadatas": metadatas})
        return out["ids"]

    def search(
        self,
        vector: List[float],
        k: int = 10,
        ef_search: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        out = self._post("/search", {
            "vector": vector, "k": k, "ef_search": ef_search, "filter": filter,
        })
        return out["results"]

    def delete(self, node_id: int) -> bool:
        resp = self._session.delete(f"{self.base_url}/delete/{node_id}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["deleted"]

    def compact(self) -> Dict[str, Any]:
        resp = self._session.post(f"{self.base_url}/compact", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        resp = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
