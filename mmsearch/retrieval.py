"""Phase 7 first-stage retrieval over the 1,215,854 US products (docs/7.md):
BM25 (SQLite FTS5, the demo's query rule and field weights), text bi-encoder (off the shelf / fine-tuned /
distilled), CLIP text->image, merged with reciprocal rank fusion."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mmsearch.esci import ROOT, RUNS
from mmsearch.text import clean

sys.path.insert(0, str(ROOT))
from app.services.pipeline import BM25_WEIGHTS, fts_query  # noqa: E402  (same rule the demo serves)

P7 = RUNS / "p7"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def corpus():
    from mmsearch.esci import load_p1_products
    p = load_p1_products()
    return p[["product_id", "product_title", "product_brand", "product_color", "product_bullet_point", "has_image"]]


def doc_text(p, bullets=300):
    f = lambda c: [clean(v, "www") for v in p[c].tolist()]                       # noqa: E731
    t, b, c, bl = f("product_title"), f("product_brand"), f("product_color"), f("product_bullet_point")
    return [f"{a} | {x} | {y} | {z[:bullets]}" for a, x, y, z in zip(t, b, c, bl)]


# ---- BM25 -----------------------------------------------------------------------------------------------------
def build_bm25(p, path=P7 / "bm25.sqlite"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE docs(rowid INTEGER PRIMARY KEY, product_id TEXT, title TEXT, brand TEXT, color TEXT, bullets TEXT);
        CREATE VIRTUAL TABLE docs_fts USING fts5(title, brand, color, bullets, content='docs', content_rowid='rowid',
                                                  tokenize='unicode61 remove_diacritics 2');""")
    rows = zip(range(1, len(p) + 1), p["product_id"], p["product_title"].fillna(""), p["product_brand"].fillna(""),
               p["product_color"].fillna(""), p["product_bullet_point"].fillna("").str.slice(0, 1000))
    db.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?)", rows)
    db.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild')")
    db.commit()
    db.close()
    return path


class BM25:
    def __init__(self, path=P7 / "bm25.sqlite"):
        self.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    def search(self, q, k=1000):
        m = fts_query(q)
        if not m:
            return [], []
        rows = self.db.execute(f"SELECT d.product_id, -bm25(docs_fts, {BM25_WEIGHTS}) s FROM docs_fts JOIN docs d "
                               f"ON d.rowid = docs_fts.rowid WHERE docs_fts MATCH ? ORDER BY s DESC LIMIT ?", (m, k)).fetchall()
        return [r[0] for r in rows], [r[1] for r in rows]


# ---- dense ----------------------------------------------------------------------------------------------------
def encode(model, texts, batch=256, query=False):
    pre = BGE_PREFIX if query else ""             # every dense model here is bge (base or fine-tuned with the prefix)
    return model.encode([pre + t for t in texts], batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float16)


def topk_ip(qv, dv, k=1000, chunk=256):
    """Exact inner-product top-k on the GPU (evaluation); the served index is FAISS HNSW (build_hnsw)."""
    import torch
    d = torch.tensor(dv, device="cuda")
    ids, scores = [], []
    for i in range(0, len(qv), chunk):
        s = torch.tensor(qv[i:i + chunk], device="cuda") @ d.T
        v, ix = s.float().topk(k, dim=1)
        ids.append(ix.cpu().numpy())
        scores.append(v.cpu().numpy())
    del d
    torch.cuda.empty_cache()
    return np.concatenate(ids), np.concatenate(scores)


def build_hnsw(dv, M=32, ef_construction=200):
    import faiss
    index = faiss.IndexHNSWFlat(dv.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(np.ascontiguousarray(dv, dtype=np.float32))
    return index


# ---- fusion ---------------------------------------------------------------------------------------------------
def rrf(lists, k=60, top=100):
    """Reciprocal rank fusion of ranked product-id lists."""
    score = {}
    for lst in lists:
        for r, pid in enumerate(lst):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + r + 1)
    return [p for p, _ in sorted(score.items(), key=lambda x: -x[1])[:top]]


def recall(ranked, relevant, k):
    rel = set(relevant)
    return len(rel & set(ranked[:k])) / len(rel) if rel else np.nan
