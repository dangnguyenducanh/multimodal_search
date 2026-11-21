"""Phase 0 §0.1: fix the splits and slices, then prove the scorer against SQID's published anchors.

Pass = random ≈ 0.7483 and SQID CLIP text 0.8107 / CLIP image 0.8225 (±0.002) on the full US Task 1 test set.
Run with the project env:  python scripts/p0_check_harness.py
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsearch.esci import RUNS, SQID, load_examples, load_products, query_slices, split, task1_us_test  # noqa: E402
from mmsearch.metrics import ndcg_per_query, paired_bootstrap  # noqa: E402
from mmsearch.runlog import write_run  # noqa: E402

ANCHORS = {"random": 0.7483, "clip-text": 0.8107, "clip-image": 0.8225}
TOL = 0.002
# Fixed pattern/material words; color words come from the catalog (see visual_words()).
PATTERN_MATERIAL = """floral striped stripe plaid polka dot dots camo camouflage paisley leopard checkered gingham
tie-dye tie dye marble glitter sequin metallic matte glossy transparent clear leather suede velvet denim wool cotton
linen silk satin lace mesh knit wooden wood bamboo metal steel stainless aluminum brass copper glass ceramic marble
plastic rubber silicone canvas nylon fleece fur faux chrome""".split()
STOP = {"and", "with", "the", "for", "pack", "set", "size", "one", "color", "colour", "assorted", "multi", "multicolor",
        "multicolored", "various", "mixed", "random", "free", "pcs", "piece", "pieces", "inch", "large", "small",
        "medium", "new", "style", "design", "print", "printed", "standard", "classic", "original", "count",
        "off", "pairs", "shown", "team", "version", "frame", "space", "oil", "tone", "colors"}


def visual_words(products):
    """Frequent words of US `product_color` values (≥ 1,000 products) + the fixed pattern/material list."""
    colors = products.loc[products["product_locale"] == "us", "product_color"].dropna().str.lower()
    counts = Counter(w for c in colors for w in set(re.findall(r"[a-z]{3,}", c)))
    return sorted({w for w, n in counts.items() if n >= 1000 and w not in STOP} | set(PATTERN_MATERIAL))


def cosine(q, p):
    return (q * p).sum(1) / (np.linalg.norm(q, axis=1) * np.linalg.norm(p, axis=1))


def main():
    out = RUNS / "p0"
    out.mkdir(parents=True, exist_ok=True)
    ex = load_examples(["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"])
    test = task1_us_test(ex).rename(columns={"esci_label": "label"})[["query_id", "query", "product_id", "label"]]

    splits = split(ex)
    splits.to_frame().to_parquet(out / "splits.parquet")
    urls = pd.read_parquet(SQID / "product_image_urls.parquet")
    test = test.merge(urls, on="product_id", how="left").assign(has_image=lambda d: d["image_url"].notna())
    words = visual_words(load_products(["product_color", "product_locale"]))
    (out / "visual_words.txt").write_text("\n".join(words) + "\n")
    slices = query_slices(test, set(words))
    slices.to_parquet(out / "slices.parquet")

    # Random: expected nDCG per query = mean over 100 shuffles.
    rng = np.random.default_rng(0)
    rand = sum(ndcg_per_query(test.assign(score=rng.random(len(test)))) for _ in range(100)) / 100

    qf = pd.read_parquet(SQID / "query_features.parquet").set_index("query_id")["clip_text_features"]
    pf = pd.read_parquet(SQID / "product_features.parquet").set_index("product_id")
    q = np.stack(qf.loc[test["query_id"]].to_numpy())
    t = np.stack(pf.loc[test["product_id"], "clip_text_features"].to_numpy())
    img = pf.loc[test["product_id"], "clip_image_features"]
    has = img.notna().to_numpy()
    i = np.zeros_like(t)
    i[has] = np.stack(img[has].to_numpy())
    s_text = cosine(q, t)
    s_img = np.where(has, cosine(q, np.where(has[:, None], i, 1.0)), -np.inf)   # no image: below every imaged product
    runs = {"random": rand,
            "clip-text": ndcg_per_query(test.assign(score=s_text)),
            "clip-image": ndcg_per_query(test.assign(score=s_img))}

    report = {}
    for name, pq in runs.items():
        rec = write_run("p0", f"anchor-{name}", pq, splits, slices, report_test=True,
                        args={"source": "SQID CLIP-L/14" if name != "random" else "mean of 100 shuffles"},
                        notes=f"scorer anchor; SQID paper reports {ANCHORS[name]} on the full US test set")
        full = float(pq.mean())
        report[name] = {"full_test": full, "expected": ANCHORS[name], "pass": abs(full - ANCHORS[name]) <= TOL,
                        "dev": rec["dev"]["ndcg"], "final_test": rec["test"]["ndcg"]}
    dev_q = splits.index[splits == "dev"]
    noise = paired_bootstrap(runs["clip-text"].reindex(dev_q), runs["clip-image"].reindex(dev_q))
    report["noise_level"] = {"pair": "clip-text vs clip-image on dev", **noise,
                             "ci_half_width": (noise["ci95"][1] - noise["ci95"][0]) / 2}
    report["counts"] = {"queries": int(test["query_id"].nunique()), "pairs": len(test),
                        "dev": int((splits == "dev").sum()), "final_test": int((splits == "final_test").sum()),
                        "visual_words": len(words), "products_without_image": int((~has).sum())}
    (out / "harness_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    ok = all(report[n]["pass"] for n in ANCHORS)
    print("HARNESS", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
