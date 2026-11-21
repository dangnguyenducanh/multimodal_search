"""retrieve → rerank → assemble (backend.md §9.5).

Today: BM25 retrieval only (SQLite FTS5). The reranker slots are deliberately empty; docs/demo-handoff.md says how
to fill them once Phases 4–6 produce a winner.
"""
import re
import sqlite3
import time
from collections.abc import Callable

from app.config import RETRIEVE_K
from app.schemas import ProductHit, SearchResponse, Timings

# Plug-in points. A reranker takes (query, hits) and returns the hits re-ordered, with score/probs/predicted filled.
Reranker = Callable[[str, list[ProductHit]], list[ProductHit]]
RERANKERS: dict[str, Reranker | None] = {"multimodal": None, "text": None}
RETRIEVER = "bm25 (sqlite fts5)"
BM25_WEIGHTS = "10.0, 4.0, 2.0, 1.0"  # title, brand, color, bullets


def norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())  # same rule as scripts/p9_build_db.py


def fts_query(q: str) -> str:
    """User text → FTS5 OR-query of quoted tokens (quoting neutralizes FTS syntax like NEAR, *, column filters)."""
    return " OR ".join(f'"{t}"' for t in re.findall(r"\w+", q.lower())[:32])


def hit(row: sqlite3.Row, score: float, label: str | None = None) -> ProductHit:
    return ProductHit(product_id=row["product_id"], title=row["title"], brand=row["brand"], color=row["color"],
                      bullets=row["bullets"], image_url=row["image_url"], score=round(score, 4), human_label=label)


def retrieve(db: sqlite3.Connection, q: str, k: int) -> list[ProductHit]:
    match = fts_query(q)
    if not match:
        return []
    rows = db.execute(
        f"SELECT p.*, -bm25(products_fts, {BM25_WEIGHTS}) AS s FROM products_fts "
        f"JOIN products p ON p.rowid = products_fts.rowid WHERE products_fts MATCH ? ORDER BY s DESC LIMIT ?",
        (match, k)).fetchall()
    return [hit(r, r["s"]) for r in rows]


def search(db: sqlite3.Connection, q: str, mode: str, k: int) -> SearchResponse:
    t0 = time.perf_counter()
    hits = retrieve(db, q, max(k, RETRIEVE_K))
    t1 = time.perf_counter()
    reranker = RERANKERS.get(mode)
    if reranker is not None:
        hits = reranker(q, hits)
    t2 = time.perf_counter()

    row = db.execute("SELECT query_id FROM queries WHERE query_norm = ?", (norm_query(q),)).fetchone()
    qid = row["query_id"] if row else None
    if qid is not None:
        labels = dict(db.execute("SELECT product_id, label FROM judgments WHERE query_id = ?", (qid,)).fetchall())
        for h in hits:
            h.human_label = labels.get(h.product_id)
    ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
    return SearchResponse(query=q, mode=mode, retriever=RETRIEVER, reranker=None if reranker is None else mode,
                          items=hits[:k], matched_query_id=qid, ground_truth=qid is not None,
                          timings_ms=Timings(retrieve=ms(t0, t1), rerank=ms(t1, t2), total=ms(t0, time.perf_counter())))


def looks_like(db: sqlite3.Connection, product_rowid: int, k: int) -> list[ProductHit]:
    row = db.execute("SELECT neighbors FROM looks_like WHERE product_rowid = ?", (product_rowid,)).fetchone()
    if row is None:
        return []
    pairs = [(int(r), float(c)) for r, c in (p.split(":") for p in row["neighbors"].split(","))][:k]
    found = {r["rowid"]: r for r in db.execute(
        f"SELECT rowid, * FROM products WHERE rowid IN ({','.join('?' * len(pairs))})", [r for r, _ in pairs])}
    return [hit(found[r], c) for r, c in pairs if r in found]
