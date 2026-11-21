"""ESCI tables and the fixed splits (docs/1.md). split() is fixed in Phase 0 and reused by every phase.

Kept importable on Python 3.9 / pandas 1.3 too: the ZR adapter (scripts/p0_zr_data.py) runs in the `zr` env.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESCI = ROOT / "data" / "esci"
SQID = ROOT / "data" / "sqid"
RUNS = ROOT / "runs"
N_DEV = 2000


def load_examples(columns=None):
    return pd.read_parquet(ESCI / "shopping_queries_dataset_examples.parquet", columns=columns)


def load_products(columns=None):
    return pd.read_parquet(ESCI / "shopping_queries_dataset_products.parquet", columns=columns)


def task1_us_test(ex):
    return ex[(ex["small_version"] == 1) & (ex["split"] == "test") & (ex["product_locale"] == "us")]


def _md5(x):
    return hashlib.md5(str(int(x)).encode()).hexdigest()


def split(ex):
    """query_id -> 'dev' | 'final_test' for the 8,956 Task 1 US test queries.

    Dev = the 2,000 query ids with the smallest md5(query_id): identical on every Python / numpy version
    (a seeded RNG is not guaranteed to be, and the zr env runs numpy 1.22 while ours runs 2.x).
    """
    qids = sorted(task1_us_test(ex)["query_id"].unique(), key=_md5)
    s = pd.Series(["dev"] * N_DEV + ["final_test"] * (len(qids) - N_DEV), index=[int(q) for q in qids], name="split")
    return s.rename_axis("query_id").sort_index()


def train_rows(ex):
    """All `train` rows (every task, every locale) minus queries that also appear in the Task 1 test set."""
    test_q = set(task1_us_test(ex)["query_id"])
    tr = ex[ex["split"] == "train"]
    return tr[~tr["query_id"].isin(test_q)]


GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def fold_of(query):
    """Stable 10-way fold of a query text (docs/1.md proposal 1): md5, so every Python version agrees."""
    return int(hashlib.md5(str(query).encode("utf8")).hexdigest(), 16) % 10


def image_urls():
    u = pd.concat([pd.read_parquet(SQID / "product_image_urls.parquet"),
                   pd.read_parquet(SQID / "supp_product_image_urls.parquet")], ignore_index=True)
    return u.dropna(subset=["image_url"]).drop_duplicates("product_id")


def load_pairs(locales=("us",), splits_=("train", "dev", "final_test")):
    """runs/p1/pairs.parquet (built by scripts/p1_build_data.py), filtered."""
    p = pd.read_parquet(RUNS / "p1" / "pairs.parquet")
    return p[p["locale"].isin(locales) & p["split"].isin(splits_)].reset_index(drop=True)


def load_p1_products(locale="us"):
    p = pd.read_parquet(RUNS / "p1" / "products.parquet")
    return p[p["product_locale"] == locale].reset_index(drop=True)


def query_slices(pairs, visual_words):
    """pairs: query_id, query, has_image (one row per judged pair) -> per-query slice labels (docs/0.md §0.1)."""
    q = pairs.groupby("query_id").agg(query=("query", "first"), all_imaged=("has_image", "all"))
    n = q["query"].str.split().str.len()
    words = q["query"].str.lower().map(lambda s: set(re.findall(r"[a-z]+", s)))
    return pd.DataFrame({
        "qlen": pd.cut(n, [0, 2, 5, 10_000], labels=["1-2", "3-5", "6+"]).astype(str),
        "visual": words.map(lambda w: "visual" if w & visual_words else "other"),
        "image": q["all_imaged"].map({True: "all_imaged", False: "some_missing"}),
    })
