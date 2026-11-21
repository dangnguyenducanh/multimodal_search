"""Build the demo database runs/p9/catalog.sqlite (backend.md §9.3).

Catalog = the US Task 1 test products (the ones judged for the 8,956 dev + final-test queries), so it is small
enough for a Vercel function. Rankers are left blank until training lands (docs/demo-handoff.md); this script only
writes data: products + FTS5 (BM25), queries, judgments, example queries, CLIP image look-alikes (SQID embeddings).

Run: python scripts/p9_build_db.py   (needs pandas, pyarrow, torch; GPU optional)
"""
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p9/catalog.sqlite"
BULLETS_MAX = 250  # chars kept per product; keeps the Vercel bundle small
LOOKALIKE_K = 8


def norm_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def main() -> None:
    ex = pd.read_parquet(ROOT / "data/esci/shopping_queries_dataset_examples.parquet")
    ex = ex[(ex.small_version == 1) & (ex.split == "test") & (ex.product_locale == "us")]
    pids = set(ex.product_id)

    prod = pd.read_parquet(ROOT / "data/esci/shopping_queries_dataset_products.parquet",
                           columns=["product_id", "product_title", "product_bullet_point",
                                    "product_brand", "product_color", "product_locale"])
    prod = prod[(prod.product_locale == "us") & prod.product_id.isin(pids)].drop(columns="product_locale")
    urls = pd.concat([pd.read_parquet(ROOT / "data/sqid/product_image_urls.parquet"),
                      pd.read_parquet(ROOT / "data/sqid/supp_product_image_urls.parquet")]).dropna()
    urls = urls.drop_duplicates("product_id")
    prod = prod.merge(urls, on="product_id", how="left").sort_values("product_id").reset_index(drop=True)
    prod["product_bullet_point"] = prod.product_bullet_point.fillna("").str.slice(0, BULLETS_MAX)
    print(f"products {len(prod):,}, with image {prod.image_url.notna().sum():,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)
    db = sqlite3.connect(OUT)
    db.executescript("""
    CREATE TABLE products(rowid INTEGER PRIMARY KEY, product_id TEXT UNIQUE NOT NULL, title TEXT, brand TEXT,
                          color TEXT, bullets TEXT, has_image INTEGER, image_url TEXT);
    CREATE VIRTUAL TABLE products_fts USING fts5(title, brand, color, bullets, content='products',
                                                 content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
    CREATE TABLE queries(query_id INTEGER PRIMARY KEY, query TEXT, query_norm TEXT, split TEXT);
    CREATE INDEX queries_norm ON queries(query_norm);
    CREATE TABLE judgments(query_id INTEGER, product_id TEXT, label TEXT, PRIMARY KEY(query_id, product_id))
        WITHOUT ROWID;
    CREATE TABLE examples(rank INTEGER PRIMARY KEY, query_id INTEGER, reason TEXT);
    CREATE TABLE models(name TEXT PRIMARY KEY, role TEXT, dev_ndcg REAL, ci_low REAL, ci_high REAL,
                        test_ndcg REAL, built_at TEXT);
    CREATE TABLE looks_like(product_rowid INTEGER PRIMARY KEY, neighbors TEXT);  -- "rowid:cos,rowid:cos,..."
    """)
    db.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?,?)",
                   [(i + 1, r.product_id, r.product_title, r.product_brand, r.product_color,
                     r.product_bullet_point, int(isinstance(r.image_url, str)),
                     r.image_url if isinstance(r.image_url, str) else None)
                    for i, r in enumerate(prod.itertuples())])
    db.execute("INSERT INTO products_fts(products_fts) VALUES('rebuild')")

    # Every test query is either dev or final test; Phase 1 (mmsearch/esci.py) assigns which. Until then: 'test'.
    q = ex[["query_id", "query"]].drop_duplicates("query_id")
    db.executemany("INSERT INTO queries VALUES (?,?,?,?)",
                   [(int(r.query_id), r.query, norm_query(r.query), "test") for r in q.itertuples()])
    db.executemany("INSERT INTO judgments VALUES (?,?,?)",
                   [(int(r.query_id), r.product_id, r.esci_label) for r in ex.itertuples()])

    # Example chips: 20 random well-imaged multi-word queries. 'gained'/'lost' examples need two trained models.
    img = set(prod.product_id[prod.image_url.notna()])
    st = ex.assign(img=ex.product_id.isin(img)).groupby(["query_id", "query"]).agg(n=("img", "size"),
                                                                                   cov=("img", "mean"))
    st = st[(st.n >= 10) & (st["cov"] >= 0.9)].reset_index()
    st = st[st["query"].str.split().str.len() >= 2]
    pick = st.sample(20, random_state=0)
    db.executemany("INSERT INTO examples VALUES (?,?,?)",
                   [(i + 1, int(r.query_id), "random") for i, r in enumerate(pick.itertuples())])

    # Look-alikes: cosine over SQID CLIP-L/14 image embeddings (no training involved).
    f = pd.read_parquet(ROOT / "data/sqid/product_features.parquet", columns=["product_id", "clip_image_features"])
    f = f[f.product_id.isin(img) & f.clip_image_features.notna()].reset_index(drop=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(np.stack(f.clip_image_features.to_numpy()), dtype=torch.float16, device=dev)
    x = torch.nn.functional.normalize(x.float(), dim=1).half()
    rowid = dict(zip(prod.product_id, prod.index + 1))
    ids, rows = f.product_id.map(rowid).to_numpy(), []
    for s in range(0, len(x), 4096):
        sim = x[s:s + 4096] @ x.T
        sim[torch.arange(sim.shape[0]), torch.arange(s, s + sim.shape[0])] = -1  # drop self
        val, idx = sim.float().topk(LOOKALIKE_K, dim=1)
        for i, (v, j) in enumerate(zip(val.cpu().numpy(), idx.cpu().numpy())):
            rows.append((int(ids[s + i]), ",".join(f"{ids[k]}:{w:.2f}" for k, w in zip(j, v))))
    db.executemany("INSERT INTO looks_like VALUES (?,?)", rows)
    print(f"queries {len(q):,}, judgments {len(ex):,}, look-alikes {len(rows):,}")

    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"{OUT} {os.path.getsize(OUT) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
