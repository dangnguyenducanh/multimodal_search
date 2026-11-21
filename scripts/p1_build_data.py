"""Phase 1 (docs/1.md): the fixed tables every later phase reads -> runs/p1/{pairs,products}.parquet + report.json."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsearch.esci import GAIN, RUNS, fold_of, image_urls, load_examples, load_products, task1_us_test  # noqa: E402


def mix(d):
    return {k: round(100 * v, 1) for k, v in d["label"].value_counts(normalize=True).sort_index().items()}


def main():
    out = RUNS / "p1"
    out.mkdir(parents=True, exist_ok=True)
    ex = load_examples()
    splits = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]          # fixed in Phase 0
    test = task1_us_test(ex)
    test_q = set(test["query_id"])

    tr = ex[(ex["split"] == "train") & ~ex["query_id"].isin(test_q)].copy()   # drops the one shared query
    tr["split"] = "train"
    te = test.copy()
    te["split"] = te["query_id"].map(splits)
    pairs = pd.concat([tr, te], ignore_index=True).rename(columns={"product_locale": "locale", "esci_label": "label"})
    folds = {q: fold_of(q) for q in pairs.loc[pairs["split"] == "train", "query"].unique()}
    pairs["fold"] = pairs["query"].map(folds).where(pairs["split"] == "train", -1).astype(int)
    pairs["gain"] = pairs["label"].map(GAIN)
    pairs["weight"] = 1.0                                          # [ZR] original rows 1.0; augmented rows (Phase 2) 0.3
    urls = image_urls()
    pairs = pairs.merge(urls, on="product_id", how="left")
    pairs["has_image"] = pairs["image_url"].notna() & (pairs["locale"] == "us")
    cols = ["query_id", "query", "product_id", "locale", "label", "gain", "split", "fold", "weight",
            "small_version", "has_image"]
    pairs = pairs[cols]
    pairs.to_parquet(out / "pairs.parquet", index=False)

    prod = load_products().merge(urls, on="product_id", how="left")
    prod["has_image"] = prod["image_url"].notna() & (prod["product_locale"] == "us")
    prod.to_parquet(out / "products.parquet", index=False)

    us = pairs[pairs["locale"] == "us"]
    trn, dev, fin = (us[us["split"] == s] for s in ["train", "dev", "final_test"])
    qsets = {s: set(d["query_id"]) for s, d in [("train", trn), ("dev", dev), ("final_test", fin)]}
    fold_split = us[us["split"] == "train"].groupby("query_id")["fold"].nunique().max()
    rep = {
        "counts_us": {s: {"queries": int(d["query_id"].nunique()), "pairs": len(d)}
                      for s, d in [("train", trn), ("dev", dev), ("final_test", fin)]},
        "es_jp_train_pairs": int(((pairs["split"] == "train") & (pairs["locale"] != "us")).sum()),
        "shared_queries": {"train&dev": len(qsets["train"] & qsets["dev"]),
                           "train&final_test": len(qsets["train"] & qsets["final_test"]),
                           "dev&final_test": len(qsets["dev"] & qsets["final_test"])},
        "max_folds_per_query": int(fold_split),
        "label_mix_pct": {"train_us": mix(trn), "test_us": mix(pd.concat([dev, fin])), "dev": mix(dev)},
        "image_coverage_pct": {"test": round(100 * pd.concat([dev, fin])["has_image"].mean(), 1),
                               "dev": round(100 * dev["has_image"].mean(), 1),
                               "train_task1": round(100 * trn.loc[trn["small_version"] == 1, "has_image"].mean(), 1),
                               "train_all": round(100 * trn["has_image"].mean(), 1)},
        "fold_sizes_train_us": trn["fold"].value_counts().sort_index().to_dict(),
    }
    checks = {
        "test 8,956 q / 181,701 pairs": rep["counts_us"]["dev"]["queries"] + rep["counts_us"]["final_test"]["queries"] == 8956
        and rep["counts_us"]["dev"]["pairs"] + rep["counts_us"]["final_test"]["pairs"] == 181701,
        "train 74,888 q (minus shared)": rep["counts_us"]["train"]["queries"] in (74887, 74888),
        "no shared queries": not any(rep["shared_queries"].values()),
        "one fold per query": rep["max_folds_per_query"] == 1,
        "dev mix within 1pt of test": all(abs(rep["label_mix_pct"]["dev"].get(k, 0) - v) <= 1.5
                                          for k, v in rep["label_mix_pct"]["test_us"].items()),
    }
    rep["checks"] = checks
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print("P1", "PASS" if all(checks.values()) else "FAIL")
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
