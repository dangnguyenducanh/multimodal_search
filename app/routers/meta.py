import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends

from app.db import get_db
from app.schemas import Example, Health, ModelInfo
from app.services import pipeline

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.get("/health", response_model=Health)
def health(db: Db):
    n = db.execute("SELECT count(*) FROM products").fetchone()[0]
    names = [r[0] for r in db.execute("SELECT name FROM models")]
    active = next((m for m, r in pipeline.RERANKERS.items() if r is not None), None)
    return Health(status="ok", db="up", products=n, models=names, reranker=active, gpu=False)


@router.get("/models", response_model=list[ModelInfo])
def models(db: Db):
    return [ModelInfo(name=r["name"], role=r["role"], dev_ndcg=r["dev_ndcg"],
                      ci=None if r["ci_low"] is None else [r["ci_low"], r["ci_high"]], test_ndcg=r["test_ndcg"])
            for r in db.execute("SELECT * FROM models ORDER BY role, name")]


@router.get("/examples", response_model=list[Example])
def examples(db: Db):
    return [Example(query=r[0], reason=r[1]) for r in db.execute(
        "SELECT q.query, e.reason FROM examples e JOIN queries q USING (query_id) ORDER BY e.rank")]
