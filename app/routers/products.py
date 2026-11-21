import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.db import get_db, product_by_id
from app.errors import ApiError
from app.schemas import HitList, Product
from app.services import pipeline

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.get("/products/{product_id}", response_model=Product)
def product(db: Db, product_id: str):
    r = product_by_id(db, product_id)
    return Product(**{k: r[k] for k in Product.model_fields})


@router.get("/products/{product_id}/looks-like", response_model=HitList)
def looks_like(db: Db, product_id: str, k: Annotated[int, Query(ge=1, le=8)] = 8):
    """Nearest products by CLIP image cosine (SQID embeddings, precomputed). Stands in for photo upload."""
    return HitList(items=pipeline.looks_like(db, product_by_id(db, product_id)["rowid"], k))


@router.get("/products/{product_id}/alternatives", response_model=HitList)
@router.get("/products/{product_id}/goes-with", response_model=HitList)
def needs_reranker(db: Db, product_id: str):
    # Ranked by the reranker's P(S) / P(C) (recommendations.md, Phase 8). Blank until a model exists.
    product_by_id(db, product_id)
    raise ApiError(503, "MODEL_NOT_READY", "needs the Phase 8 reranker; not trained yet")
