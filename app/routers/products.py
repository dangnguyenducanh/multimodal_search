import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.db import get_db, has_table, product_by_id
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


def recs(db: sqlite3.Connection, product_id: str, kind: str, k: int) -> HitList:
    """Ranked by the reranker's P(S) / P(C) (docs/8.md), precomputed by scripts/p9_add_models.py for the products in
    the example queries' judged lists; other products get an empty list."""
    product_by_id(db, product_id)
    if not has_table(db, "recs"):
        raise ApiError(503, "MODEL_NOT_READY", "needs the Phase 8 reranker; not trained yet")
    rows = db.execute("""SELECT p.*, r.score FROM recs r JOIN products p ON p.product_id = r.rec_product_id
                         WHERE r.product_id = ? AND r.kind = ? ORDER BY r.rank LIMIT ?""", (product_id, kind, k))
    return HitList(items=[pipeline.hit(r, r["score"]) for r in rows])


@router.get("/products/{product_id}/alternatives", response_model=HitList)
def alternatives(db: Db, product_id: str, k: Annotated[int, Query(ge=1, le=10)] = 10):
    return recs(db, product_id, "alternatives", k)


@router.get("/products/{product_id}/goes-with", response_model=HitList)
def goes_with(db: Db, product_id: str, k: Annotated[int, Query(ge=1, le=10)] = 10):
    return recs(db, product_id, "goes_with", k)
