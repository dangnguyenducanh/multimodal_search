import sqlite3
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query

from app.db import get_db
from app.errors import ApiError
from app.schemas import SearchResponse
from app.services import pipeline

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]


@router.get("/search", response_model=SearchResponse)
def search(db: Db, q: Annotated[str, Query(min_length=1, max_length=200)],
           mode: Literal["multimodal", "text"] = "multimodal", k: Annotated[int, Query(ge=1, le=50)] = 20):
    if not q.strip():
        raise ApiError(400, "VALIDATION_ERROR", "q must be 1-200 characters", [{"path": "q", "issue": "blank"}])
    return pipeline.search(db, q, mode, k)


@router.post("/search/image")
def search_image():
    # Needs a CLIP image encoder at request time: too big for a Vercel function. See docs/demo-handoff.md.
    raise ApiError(503, "MODEL_NOT_READY", "photo search is not deployed in this demo")
