"""Phase 2 data report (docs/2.md): truncation per length/format/field, empty fields, augmentation sizes.

  python scripts/p2_preprocess.py            # -> runs/p2/report.json (value checks run through the proxy ladder)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsearch.augment import rebalance  # noqa: E402
from mmsearch.esci import RUNS, load_p1_products, load_pairs  # noqa: E402
from mmsearch.text import ORDERS, product_fields  # noqa: E402

TOKENIZER = "FacebookAI/xlm-roberta-base"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    pairs = load_pairs(splits_=("train", "dev"))
    prod = load_p1_products().set_index("product_id")
    sample = pairs.sample(20_000, random_state=0)
    rep = {"tokenizer": TOKENIZER, "sample_pairs": len(sample)}
    for how in ["zr", "www"]:
        f = product_fields(prod.loc[sample["product_id"]].reset_index(drop=True), how=how)
        q = tok(sample["query"].tolist(), add_special_tokens=False)["input_ids"]
        lens = {c: np.array([len(x) for x in tok(f[c].tolist(), add_special_tokens=False)["input_ids"]])
                for c in ORDERS["default"]}
        qlen = np.array([len(x) for x in q])
        r = {"empty_pct": {c: round(100 * (f[c] == "").mean(), 1) for c in ORDERS["default"]},
             "median_tokens": {c: int(np.median(v)) for c, v in lens.items()}, "query_median_tokens": int(np.median(qlen))}
        for L in [128, 256, 384]:
            # format B, default order: <s> q </s></s> f1 </s></s> f2 ... </s>  (2 special tokens per separator)
            budget = L - qlen - 4
            pos = np.zeros(len(sample))
            cut = {}
            for c in ORDERS["default"]:
                start = pos.copy()
                pos = pos + lens[c] + 2
                cut[c] = round(100 * ((start >= budget) & (lens[c] > 0)).mean(), 1)        # field fully lost
            r[f"L{L}"] = {"pairs_truncated_pct": round(100 * (pos - 2 > budget).mean(), 1),
                          "field_fully_lost_pct": cut,
                          "query_cut_with_longest_first_pct": round(100 * (qlen > L // 2).mean(), 2)}
        rep[f"clean_{how}"] = r
    fields = product_fields(prod.loc[pairs.loc[pairs["split"] == "train", "product_id"].unique()])
    reb = rebalance(pairs, fields)
    rep["rebalance_rows"] = {"added": len(reb), "of_train": int((pairs["split"] == "train").sum())}
    out = RUNS / "p2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
