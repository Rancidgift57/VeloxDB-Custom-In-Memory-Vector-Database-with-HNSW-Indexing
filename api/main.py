"""
api/main.py
FastAPI gateway. Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Configuration via environment variables (see docker-compose.yml):
    VDB_DIM, VDB_METRIC, VDB_M, VDB_EF_CONSTRUCTION, VDB_EF_SEARCH, VDB_DATA_DIR
"""
from __future__ import annotations
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.database import VectorDatabase
from api.schemas import (
    InsertRequest, InsertResponse,
    InsertBatchRequest, InsertBatchResponse,
    SearchRequest, SearchResponse, SearchHit,
    DeleteResponse, HealthResponse,
)

DIM = int(os.environ.get("VDB_DIM", "128"))
METRIC = os.environ.get("VDB_METRIC", "cosine")
M = int(os.environ.get("VDB_M", "16"))
EF_CONSTRUCTION = int(os.environ.get("VDB_EF_CONSTRUCTION", "200"))
EF_SEARCH = int(os.environ.get("VDB_EF_SEARCH", "50"))
DATA_DIR = os.environ.get("VDB_DATA_DIR", "./data")
QUANTIZE = os.environ.get("VDB_QUANTIZE", "false").lower() == "true"

app = FastAPI(title="MiniVectorDB", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

db = VectorDatabase(
    dim=DIM, metric=METRIC, M=M,
    ef_construction=EF_CONSTRUCTION, ef_search=EF_SEARCH, data_dir=DATA_DIR,
    quantize=QUANTIZE,
)


@app.post("/fit_quantizer")
def fit_quantizer(req: InsertBatchRequest):
    """Required exactly once, before any /insert calls, when the server was
    started with VDB_QUANTIZE=true. Fits the int8 quantizer's per-dimension
    min/max range on a representative sample (a few hundred vectors is
    usually enough) before storage switches to 4x-smaller int8 rows."""
    if not QUANTIZE:
        raise HTTPException(400, "server was not started with VDB_QUANTIZE=true")
    for v in req.vectors:
        if len(v) != DIM:
            raise HTTPException(400, f"all vectors must have dim={DIM}")
    db.fit_quantizer(req.vectors)
    return {"status": "quantizer fitted", "sample_size": len(req.vectors)}


@app.post("/insert", response_model=InsertResponse)
def insert(req: InsertRequest):
    if len(req.vector) != DIM:
        raise HTTPException(400, f"vector must have dim={DIM}, got {len(req.vector)}")
    if QUANTIZE and not db.index.quantizer.fitted:
        raise HTTPException(409, "call /fit_quantizer with a sample batch before inserting")
    node_id = db.insert(req.vector, req.metadata)
    return InsertResponse(id=node_id)


@app.post("/insert_batch", response_model=InsertBatchResponse)
def insert_batch(req: InsertBatchRequest):
    for v in req.vectors:
        if len(v) != DIM:
            raise HTTPException(400, f"all vectors must have dim={DIM}")
    if QUANTIZE and not db.index.quantizer.fitted:
        raise HTTPException(409, "call /fit_quantizer with a sample batch before inserting")
    ids = db.insert_batch(req.vectors, req.metadatas)
    return InsertBatchResponse(ids=ids)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if len(req.vector) != DIM:
        raise HTTPException(400, f"vector must have dim={DIM}, got {len(req.vector)}")
    start = time.perf_counter()
    results = db.search(req.vector, k=req.k, ef_search=req.ef_search, filter=req.filter)
    elapsed_ms = (time.perf_counter() - start) * 1000
    hits = [SearchHit(**r) for r in results]
    return SearchResponse(results=hits, query_time_ms=elapsed_ms)


@app.delete("/delete/{node_id}", response_model=DeleteResponse)
def delete(node_id: int):
    deleted = db.delete(node_id)
    return DeleteResponse(id=node_id, deleted=deleted)


@app.post("/compact")
def compact():
    db.compact()
    return {"status": "compacted", "stats": db.stats()}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", stats=db.stats())


@app.on_event("shutdown")
def shutdown_event():
    db.close()
