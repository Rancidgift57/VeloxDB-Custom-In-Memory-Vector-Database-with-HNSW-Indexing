from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class InsertRequest(BaseModel):
    vector: List[float] = Field(..., description="Embedding vector")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("vector must not be empty")
        return v


class InsertBatchRequest(BaseModel):
    vectors: List[List[float]]
    metadatas: Optional[List[Dict[str, Any]]] = None


class InsertResponse(BaseModel):
    id: int


class InsertBatchResponse(BaseModel):
    ids: List[int]


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = Field(default=10, ge=1, le=1000)
    ef_search: Optional[int] = Field(default=None, ge=1)
    filter: Optional[Dict[str, Any]] = None


class SearchHit(BaseModel):
    id: int
    distance: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchHit]
    query_time_ms: float


class DeleteResponse(BaseModel):
    id: int
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    stats: Dict[str, Any]
