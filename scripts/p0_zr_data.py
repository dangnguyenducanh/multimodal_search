"""Phase 0 §0.3: ESCI parquet -> the CSV files ZhichunRoad's scripts expect (runs/p0/zr_data/).

Run in the `zr` env with their hash seed (their 1_query_hash.py sets it the same way):
  PYTHONHASHSEED=12345 PYTHONNOUSERSITE=1 ~/miniconda3/envs/zr/bin/python scripts/p0_zr_data.py --folds 01
  ... --folds 23 45 67 89     # extra out-of-fold sets for their confident-learning filter (written when needed)
  ... --cf-filter              # after their filter: train_01_cf.csv = train_01 minus the flagged rows (member r2)
"""
import argparse
import hashlib
import os
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsearch.esci import RUNS, load_examples, load_products, task1_us_test, train_rows  # noqa: E402

OUT = RUNS / "p0" / "zr_data"
CF = OUT / "extra_confident_learnging"            # their directory name, typo included
LABEL_WORD = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}
COLS = ["query_id", "query", "query_locale", "product_id", "esci_label", "product_title", "product_brand",
        "product_color_name", "product_bullet_point", "product_description"]
MAX_CHARS = 4000     # per text field; ≥ 256 tokens of any language, so the 256-token inputs are unchanged
N_VALID_Q = 2000     # their validation file, capped (docs/0.md §0.3)


def md5(x):
    return hashlib.md5(str(int(x)).encode()).hexdigest()


def product_table():
    p = load_products().rename(columns={"product_color": "product_color_name"})
    for c in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color_name"]:
        p[c] = p[c].str.slice(0, MAX_CHARS)
    return p


def their_format(ex, products):
    d = ex.rename(columns={"product_locale": "query_locale"})
    d["esci_label"] = d["esci_label"].map(LABEL_WORD)
    d = d.merge(products, how="left", left_on=["query_locale", "product_id"], right_on=["product_locale", "product_id"])
    return d[COLS]


def write_vocab(products):
    """brand.txt / color.txt exactly as their 3_create_pretrain_data_brand_color.py builds them: values seen more than
    once in the whole catalogue, 3 [UNK] lines, then Python-set order (fixed by PYTHONHASHSEED=12345). Their models
    hard-code the table sizes (RobertaWithSampleWeight.py: brand 151,519, color 47,036), so a larger vocab would
    index past the embedding table."""
    clean = str.maketrans("\n", " ", string.punctuation)
    for col, name, size in [("product_brand", "brand", 151519), ("product_color", "color", 47036)]:
        vals = [str(v).translate(clean).lower().strip() for v in products[col].fillna("").tolist()
                if v and not v.isspace()]
        counts = {}
        for v in vals:
            if v and not v.isspace():
                counts[v] = counts.get(v, 0) + 1
        vocab = [v for v in set(k for k, n in counts.items() if n > 1) if v and not v.isspace()]
        lines = ["[UNK]"] * 3 + vocab
        print(f"{name}.txt: {len(lines):,} lines (their model's table: {size:,})")
        if len(lines) > size:
            sys.exit(f"{name} vocab larger than their embedding table; decide how to cut it before training")
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf8")


def capped(d, n=N_VALID_Q):
    keep = sorted(d["query_id"].unique(), key=md5)[:n]
    return d[d["query_id"].isin(keep)]


def write(d, path):
    d.to_csv(path, index=False, encoding="utf8")
    print(f"{path.relative_to(RUNS)}: {len(d):,} rows, {d['query_id'].nunique():,} queries, "
          f"{path.stat().st_size / 1e9:.2f} GB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", default=[], help="2-part folds to write, e.g. 01 23")
    ap.add_argument("--cf-filter", action="store_true")
    ap.add_argument("--vocab", action="store_true", help="only rebuild brand.txt / color.txt")
    a = ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "12345" or sys.version_info[:2] != (3, 9):
        sys.exit("run with PYTHONHASHSEED=12345 on Python 3.9 (their fold split uses the built-in str hash)")
    CF.mkdir(parents=True, exist_ok=True)
    if a.vocab:
        write_vocab(load_products(["product_brand", "product_color"]))
        return

    if a.cf_filter:
        issues = pd.read_csv(CF / "label_issue_index.csv")["label_issue_index"]
        oof = pd.concat([pd.read_csv(CF / f"flod_{f}_valid.csv", usecols=["query_id", "product_id"])
                         for f in ["01", "23", "45", "67", "89"]], ignore_index=True)
        flagged = oof.iloc[issues.to_numpy()]
        tr = pd.read_csv(OUT / "train_01.csv")
        key = tr["query_id"].astype(str) + "|" + tr["product_id"]
        drop = set(flagged["query_id"].astype(str) + "|" + flagged["product_id"])
        print(f"confident learning flagged {len(flagged):,} rows; {key.isin(drop).sum():,} are in train_01")
        write(tr[~key.isin(drop)], OUT / "train_01_cf.csv")
        return

    ex = load_examples(["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"])
    products = product_table()

    if not (OUT / "catalogue_test.csv").exists():
        test = task1_us_test(ex)
        splits = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
        for name in ["dev", "final_test"]:
            q = splits.index[splits == name]
            write(their_format(test[test["query_id"].isin(q)], products)[COLS[:5]], OUT / f"{name}.csv")
        cat = products[products["product_id"].isin(test["product_id"]) & (products["product_locale"] == "us")]
        write(cat.assign(query_id=0), OUT / "catalogue_test.csv")
        write_vocab(load_products(["product_brand", "product_color"]))

    if a.folds:
        tr = train_rows(ex)
        part = tr["query"].map(lambda q: hash(str(q)) % 10)   # their 1_query_hash.py
        tr = their_format(tr, products).assign(part=part.to_numpy())
        tr = tr.sample(frac=1, random_state=12345).reset_index(drop=True)
        print("parts:", tr["part"].value_counts().sort_index().to_dict())
        for f in a.folds:
            held = tr["part"].isin([int(f[0]), int(f[1])])
            write(tr.loc[~held, COLS], OUT / f"train_{f}.csv")
            write(capped(tr.loc[held, COLS]), OUT / f"valid_{f}_2k.csv")
            write(tr.loc[held, COLS], CF / f"flod_{f}_valid.csv")


if __name__ == "__main__":
    main()
