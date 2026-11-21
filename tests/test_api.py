"""Contract tests for the demo API (backend.md §9.8). CPU only.
test_api: smoke test on the real runs/p9/catalog.sqlite (read-only), whatever models it holds yet.
test_models_*: a tiny temporary DB (the real DB's schema, 4 made-up products) whose model tables are written by
scripts/p9_add_models.py's own writers."""
import importlib.util
import math
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import db as app_db
from app.main import app

ROOT = Path(__file__).resolve().parents[1]
c = TestClient(app)


def test_api():
    assert c.get("/api/health").json()["status"] == "ok"
    ex = c.get("/api/examples").json()
    assert 1 <= len(ex) <= 20

    r = c.get("/api/search", params={"q": ex[0]["query"]}).json()
    assert r["ground_truth"] and r["items"]
    assert any(h["human_label"] for h in r["items"])
    assert c.get("/api/search", params={"q": "zzzqqq xxjjkw"}).json()["items"] == []
    assert c.get("/api/search", params={"q": 'a" OR NEAR(*'}).status_code == 200  # FTS syntax neutralized

    for bad in ("", " ", "x" * 201):
        r = c.get("/api/search", params={"q": bad})
        assert r.status_code == 400 and r.json()["error"]["code"] == "VALIDATION_ERROR"

    pid = c.get("/api/search", params={"q": ex[0]["query"]}).json()["items"][0]["product_id"]
    assert c.get(f"/api/products/{pid}").json()["product_id"] == pid
    assert c.get(f"/api/products/{pid}/looks-like").status_code == 200
    assert c.get(f"/api/products/{pid}/alternatives").status_code in (200, 503)  # 503 until recs exist
    assert c.get("/api/products/NOPE").json()["error"]["code"] == "PRODUCT_NOT_FOUND"
    assert c.get("/api/nope").status_code == 404
    assert isinstance(c.get("/api/models").json(), list)


def test_app_imports_without_numpy():  # the Vercel bundle ships fastapi only
    code = "import sys, app.main; assert not {'numpy', 'pandas', 'torch'} & set(sys.modules), sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# ---- tiny DB with fake models --------------------------------------------------------------------------------
P = ["B000000001", "B000000002", "B000000003", "B000000004"]
LABELS = dict(zip(P, "ESCI"))
TITLES = ["red running shoes", "blue running shoes", "running shoe laces", "garden hose"]
ZR = [0.6, 0.7, 0.8, 0.9]      # ranks I, C, S, E: worst possible
OURS = [0.9, 0.5, 0.5, 0.1]    # ideal order; P2/P3 tie -> product_id
MM = [0.2, 0.9, 0.5, 0.1]      # multimodal: P2, P3, P1, P4


def _p9():
    spec = importlib.util.spec_from_file_location("p9_add_models", ROOT / "scripts" / "p9_add_models.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pairs(scores, probs=True):
    d = pd.DataFrame({"query_id": 1, "product_id": P, "score": scores})
    for i, k in enumerate(["pE", "pS", "pC", "pI"]):
        d[k] = [0.7 if j == i else 0.1 for j in range(4)] if probs else float("nan")
    return d


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    path = tmp_path_factory.mktemp("p9") / "catalog.sqlite"
    src = sqlite3.connect(f"file:{app_db.db_file()}?mode=ro", uri=True)
    db = sqlite3.connect(path)
    for (sql,) in src.execute("SELECT sql FROM sqlite_master WHERE name IN "
                              "('products', 'products_fts', 'queries', 'judgments', 'examples', 'models')"):
        db.execute(sql)
    src.close()
    db.executemany("INSERT INTO products(rowid, product_id, title, has_image) VALUES (?,?,?,0)",
                   [(i + 1, p, t) for i, (p, t) in enumerate(zip(P, TITLES))])
    db.execute("INSERT INTO products_fts(products_fts) VALUES('rebuild')")
    db.executemany("INSERT INTO queries VALUES (?,?,?,?)",
                   [(1, "Running  Shoes", "running shoes", "test"), (2, "garden hose", "garden hose", "test")])
    db.executemany("INSERT INTO judgments VALUES (?,?,?)", [(1, p, lab) for p, lab in LABELS.items()]
                   + [(2, P[3], "E")])
    m = _p9()
    m.prepare(db)
    m.write_model(db, "p0/zr-fake", "zr", _pairs(ZR), {"dev": {"ndcg": 0.5, "ci95": [0.4, 0.6]}, "ms_per_query": 900.0})
    m.write_model(db, "p4/old", "ours", _pairs(OURS), {})
    m.write_model(db, "p4/ours-fake", "ours", _pairs(OURS, probs=False), {"dev": {"ndcg": 0.9}, "ms_per_query": 12.5})
    m.write_model(db, "p5/mm-fake", "multimodal", _pairs(MM), {"dev": {"ndcg": 0.95}})
    db.executemany("INSERT INTO recs VALUES (?,?,?,?,?)",
                   [(P[0], "alternatives", 1, P[1], 0.8), (P[0], "alternatives", 2, P[3], 0.1),
                    (P[0], "goes_with", 1, P[2], 0.6)])
    db.commit()
    db.close()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app_db, "DB_PATH", path)
        yield c


def test_models_search(tiny):
    r = tiny.get("/api/search", params={"q": "running shoes", "mode": "text"}).json()
    assert r["reranker"] == "p4/ours-fake" and r["matched_query_id"] == 1
    assert [h["product_id"] for h in r["items"]] == P                      # judged list, ties by product_id
    assert [h["human_label"] for h in r["items"]] == list("ESCI")
    assert r["items"][0]["probs"] is None and r["items"][0]["rank_text_only"] is None   # regression-only model

    r = tiny.get("/api/search", params={"q": "RUNNING shoes "}).json()       # multimodal, normalised query
    assert r["reranker"] == "p5/mm-fake"
    assert [(h["product_id"], h["rank_text_only"]) for h in r["items"]] == [(P[1], 2), (P[2], 3), (P[0], 1), (P[3], 4)]
    assert r["items"][0]["probs"] == {"E": 0.1, "S": 0.7, "C": 0.1, "I": 0.1} and r["items"][0]["predicted"] == "S"

    for q in ("garden hose", "shoes"):                                      # no scores / free text: BM25 only
        r = tiny.get("/api/search", params={"q": q}).json()
        assert r["reranker"] is None and r["items"]


def test_models_compare(tiny):
    r = tiny.get("/api/compare", params={"q": "running shoes"})
    assert r.status_code == 200
    r = r.json()
    assert r["query"] == "Running  Shoes" and r["query_id"] == 1
    assert r["models"] == {"zr": "p0/zr-fake", "ours": "p4/ours-fake"}
    assert [(i["product_id"], i["human_label"], i["zr_rank"], i["ours_rank"]) for i in r["items"]] == [
        (P[0], "E", 4, 1), (P[1], "S", 3, 2), (P[2], "C", 2, 3), (P[3], "I", 1, 4)]
    assert r["items"][0]["zr_score"] == 0.6 and r["items"][0]["ours_score"] == 0.9
    # by hand: ZR ranks I, C, S, E; ideal is E, S, C, I; gains E 1, S 0.1, C 0.01, I 0; discount 1 / log2(rank + 1)
    idcg = 1 + 0.1 / math.log2(3) + 0.01 / math.log2(4)
    assert r["ndcg"]["zr"] == pytest.approx((0.01 / math.log2(3) + 0.1 / math.log2(4) + 1 / math.log2(5)) / idcg)
    assert r["ndcg"]["zr"] == pytest.approx(0.455940, abs=1e-6) and r["ndcg"]["ours"] == pytest.approx(1.0)

    for q in ("unknown query", "garden hose"):                              # not a query / no saved scores
        r = tiny.get("/api/compare", params={"q": q})
        assert r.status_code == 404 and r.json()["error"]["code"] == "NO_BASELINE_FOR_QUERY"


def test_models_recs_meta(tiny):
    alt = tiny.get(f"/api/products/{P[0]}/alternatives").json()["items"]
    assert [(h["product_id"], h["score"]) for h in alt] == [(P[1], 0.8), (P[3], 0.1)]
    assert [h["product_id"] for h in tiny.get(f"/api/products/{P[0]}/alternatives", params={"k": 1}).json()["items"]] == [P[1]]
    assert [h["product_id"] for h in tiny.get(f"/api/products/{P[0]}/goes-with").json()["items"]] == [P[2]]
    assert tiny.get(f"/api/products/{P[1]}/goes-with").json() == {"items": []}
    assert tiny.get("/api/products/NOPE/alternatives").status_code == 404

    h = tiny.get("/api/health").json()
    assert h["reranker"] == "p5/mm-fake" and sorted(h["models"]) == ["p0/zr-fake", "p4/ours-fake", "p5/mm-fake"]
    models = {m["name"]: m for m in tiny.get("/api/models").json()}
    assert set(models) == {"p0/zr-fake", "p4/ours-fake", "p5/mm-fake"}                 # p4/old replaced
    assert models["p0/zr-fake"] == {"name": "p0/zr-fake", "role": "zr", "dev_ndcg": 0.5, "ci": [0.4, 0.6],
                                    "test_ndcg": None, "ms_per_query": 900.0}
    assert models["p4/ours-fake"]["ms_per_query"] == 12.5 and models["p5/mm-fake"]["ms_per_query"] is None
