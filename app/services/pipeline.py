"""retrieve → rerank → assemble (backend.md §9.5).

Retrieval: BM25 (SQLite FTS5). Rerankers (app/services/rerank.py): precomputed model scores for known dev/test
queries (any host), or a live model on a GPU host; docs/demo-handoff.md says how to plug them in.
"""
import re
import sqlite3
import time

from app.config import RETRIEVE_K
from app.schemas import ProductHit, SearchResponse, Timings
from app.services import rerank

# Plug-in points: r(db, query, hits, query_id) -> hits | None, and r.model(db) -> model name (see rerank.py).
RERANKERS: dict = {"multimodal": None, "text": None}
rerank.install(RERANKERS)
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


def query_id(db: sqlite3.Connection, q: str) -> int | None:
    row = db.execute("SELECT query_id FROM queries WHERE query_norm = ?", (norm_query(q),)).fetchone()
    return row[0] if row else None


def search(db: sqlite3.Connection, q: str, mode: str, k: int) -> SearchResponse:
    t0 = time.perf_counter()
    hits = retrieve(db, q, max(k, RETRIEVE_K))
    t1 = time.perf_counter()
    qid = query_id(db, q)
    r, name = RERANKERS.get(mode), None
    ranked = r(db, q, hits, qid) if r is not None else None
    if ranked is not None:
        name = r.model(db)
        t = RERANKERS.get("text")
        if mode == "multimodal" and t is not None and t.model(db) != name:  # the toggle's rank arrows
            text = t(db, q, hits, qid)
            if text is not None:
                pos = {h.product_id: i + 1 for i, h in enumerate(text)}
                for h in ranked:
                    h.rank_text_only = pos.get(h.product_id)
        hits = ranked
    t2 = time.perf_counter()

    if qid is not None:
        labels = dict(db.execute("SELECT product_id, label FROM judgments WHERE query_id = ?", (qid,)).fetchall())
        for h in hits:
            h.human_label = labels.get(h.product_id)
    ms = lambda a, b: round((b - a) * 1000, 1)  # noqa: E731
    return SearchResponse(query=q, mode=mode, retriever=RETRIEVER, reranker=name,
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
