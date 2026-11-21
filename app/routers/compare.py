"""GET /api/compare (docs/9.md): a known dev/test query's judged products ranked by ZR's ensemble and by ours, from
the saved scores in pair_scores (no ZR model at serve time)."""
import math
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.db import get_db
from app.errors import ApiError
from app.schemas import CompareItem, CompareModels, CompareNdcg, CompareResponse, Product
from app.services import pipeline
from app.services.rerank import model_for

router = APIRouter()
Db = Annotated[sqlite3.Connection, Depends(get_db)]
GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}  # mmsearch/metrics.py, in pure Python (no numpy on Vercel)


def ndcg(labels: list[str]) -> float:
    """nDCG of the whole judged list in ranked order; 0 when nothing is relevant."""
    dcg = lambda gains: sum(g / math.log2(i + 2) for i, g in enumerate(gains))  # noqa: E731
    g = [GAIN[x] for x in labels]
    ideal = dcg(sorted(g, reverse=True))
    return dcg(g) / ideal if ideal > 0 else 0.0


@router.get("/compare", response_model=CompareResponse)
def compare(db: Db, q: Annotated[str, Query(min_length=1, max_length=200)]):
    qid, zr, ours = pipeline.query_id(db, q), model_for(db, "zr"), model_for(db, "ours")
    rows = [] if qid is None or zr is None or ours is None else db.execute("""
        SELECT p.*, j.label, z.score AS zr_score, o.score AS ours_score FROM judgments j
        JOIN products p ON p.product_id = j.product_id
        JOIN pair_scores z ON z.query_id = j.query_id AND z.product_id = j.product_id AND z.model = ?
        JOIN pair_scores o ON o.query_id = j.query_id AND o.product_id = j.product_id AND o.model = ?
        WHERE j.query_id = ?""", (zr, ours, qid)).fetchall()
    if not rows:
        raise ApiError(404, "NO_BASELINE_FOR_QUERY", "not a dev/test query with saved ZR and model scores")
    by = lambda col: sorted(rows, key=lambda r: (-r[col], r["product_id"]))  # noqa: E731  ties: product_id
    zr_order, ours_order = by("zr_score"), by("ours_score")
    zr_rank = {r["product_id"]: i + 1 for i, r in enumerate(zr_order)}
    items = [CompareItem(**{k: r[k] for k in Product.model_fields}, human_label=r["label"],
                         zr_rank=zr_rank[r["product_id"]], ours_rank=i + 1, zr_score=r["zr_score"],
                         ours_score=r["ours_score"]) for i, r in enumerate(ours_order)]
    query = db.execute("SELECT query FROM queries WHERE query_id = ?", (qid,)).fetchone()[0]
    return CompareResponse(query=query, query_id=qid, models=CompareModels(zr=zr, ours=ours), items=items,
                           ndcg=CompareNdcg(zr=ndcg([r["label"] for r in zr_order]),
                                            ours=ndcg([r["label"] for r in ours_order])))
