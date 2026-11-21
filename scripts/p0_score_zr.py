"""Phase 0 §0.6: score the ZR members and ensembles with OUR nDCG; check zr_score() against THEIR ranking file.

Rows (only those whose members exist): c1, r1, c1+r1, c1+c2+c3 (their sort_key_3), c1-c3+r1, and the baseline
c1-c3+r1+r2 (their sort_key_4). The baseline's headline number is the nDCG of their own ranking file.
  python scripts/p0_score_zr.py                      # writes runs/p0/zr-*/metrics.json
  python scripts/p0_score_zr.py --pred runs/p0/zr_pred_test --no-write   # test on a subset
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsearch.ensemble import ZR_LABEL, zr_score  # noqa: E402
from mmsearch.esci import RUNS, load_examples, task1_us_test  # noqa: E402
from mmsearch.metrics import micro_f1, ndcg_per_query  # noqa: E402
from mmsearch.runlog import BASELINE_RUN, write_run  # noqa: E402

ROWS = [  # (run name, classifiers, regressors, their public-LB log)
    ("zr-c123-r12", ["c1", "c2", "c3"], ["r1", "r2"], 0.9032),   # first, so the others get vs_baseline
    ("zr-c1", ["c1"], [], None),
    ("zr-r1", [], ["r1"], 0.9001),
    ("zr-c1-r1", ["c1"], ["r1"], None),
    ("zr-c123", ["c1", "c2", "c3"], [], 0.8946),
    ("zr-c123-r1", ["c1", "c2", "c3"], ["r1"], None),
]
DEVIATIONS = ("start xlm-roberta-large (their pretrained checkpoint unreleased); 1 epoch not 10; fp16 (accelerate "
              "MIXED_PRECISION) not fp32; one 10-part hash split; no translation/rebalancing; validation capped at "
              "2,000 queries; torch 2.0.1+cu118; 4-class model class rebuilt from their eval copy (ce-poly-1)")


def ranked_like_them(df, score):
    """Their ranking: per query, sort by score descending; Python's stable sort keeps input order on ties."""
    order = np.lexsort((np.arange(len(df)), -score, df["query_id"].to_numpy()))
    return df.iloc[order][["query_id", "product_id"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(RUNS / "p0" / "zr_pred"))
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    pred = Path(a.pred)
    ex = load_examples(["query_id", "product_id", "product_locale", "esci_label", "small_version", "split"])
    labels = task1_us_test(ex)[["query_id", "product_id", "esci_label"]].rename(columns={"esci_label": "label"})
    splits = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
    slices = pd.read_parquet(RUNS / "p0" / "slices.parquet")
    models = RUNS / "p0" / "zr_models"
    train_s = {m: json.loads((models / m / "train_time.json").read_text())["seconds"]
               for m in ["c1", "c2", "c3", "r1", "r2", "f23", "f45", "f67", "f89"]
               if (models / m / "train_time.json").exists()}
    timing = json.loads((pred / "timing.json").read_text()) if (pred / "timing.json").exists() else {}

    per_split = {}
    for split in ["dev", "final_test"]:
        outs = {f.stem.split("_")[1]: pd.read_parquet(f) for f in pred.glob(f"member_*_{split}.parquet")}
        if not outs:
            continue
        base = next(iter(outs.values()))[["query_id", "product_id"]]
        for m, o in outs.items():
            assert o[["query_id", "product_id"]].equals(base), f"{m} {split}: rows not aligned"
        df = base.merge(labels, on=["query_id", "product_id"], how="left", validate="1:1")
        assert df["label"].notna().all() and df.index.equals(base.index)
        per_split[split] = (df, {m: o.filter(like="s").to_numpy() for m, o in outs.items()})

    report = {}
    for name, cs, rs, logged in ROWS:
        if not all(m in per_split.get("dev", (None, {}))[1] for m in cs + rs):
            continue
        pq, parity = [], {}
        for split, (df, out) in per_split.items():
            if not all(m in out for m in cs + rs):
                continue
            score = zr_score([out[m] for m in cs], [out[m] for m in rs])
            pq.append(ndcg_per_query(df.assign(score=score)))
            if name == BASELINE_RUN[1] and (pred / f"ranking_{split}.csv").exists():
                theirs = pd.read_csv(pred / f"ranking_{split}.csv")
                ours = ranked_like_them(df, score)
                t = theirs.groupby("query_id")["product_id"].agg(list)
                o = ours.groupby("query_id")["product_id"].agg(list)
                same = t.index.equals(o.index) and all(t[q] == o[q] for q in t.index)
                pos = -theirs.groupby("query_id").cumcount().to_numpy(dtype=float)   # their order as a score
                theirs_ndcg = ndcg_per_query(theirs.merge(labels, on=["query_id", "product_id"]).assign(score=pos))
                parity[split] = {"identical_order": bool(same), "ndcg_their_file": float(theirs_ndcg.mean()),
                                 "ndcg_zr_score": float(pq[-1].mean())}
                if not same:
                    raise SystemExit(f"zr_score() does not reproduce their ranking on {split}")
                # per-query numbers use zr_score with the project's tie-break (product_id), like every other run;
                # their file breaks exact ties by input order, so the two can differ only on exact ties (logged).
        per_query = pd.concat(pq)
        members = cs + rs + (["f23", "f45", "f67", "f89"] if "r2" in rs else [])
        gpu_h = sum(train_s.get(m, 0) for m in members) / 3600
        dev_s = sum(timing.get(f"{m}_dev_s", 0) for m in cs + rs)
        dev_df, dev_out = per_split["dev"]
        extra = {}
        if cs:
            p = np.mean([dev_out[m] for m in cs], axis=0)
            pred_lab = ZR_LABEL[p.argmax(1)]
            extra = {"task2_micro_f1": micro_f1(pred_lab, dev_df["label"]),
                     "task3_substitute_f1": micro_f1(pred_lab == "S", dev_df["label"] == "S")}
        row = {"dev_ndcg": float(per_query.reindex(splits.index[splits == "dev"]).mean()), "their_public_lb": logged,
               "parity": parity, "gpu_hours": round(gpu_h, 2), **extra}
        report[name] = row
        print(name, json.dumps(row), flush=True)
        if not a.no_write:
            full = "final_test" in per_split and all(m in per_split["final_test"][1] for m in cs + rs)
            write_run("p0", name, per_query, splits, slices, report_test=(name == BASELINE_RUN[1] and full),
                      args={"classifiers": cs, "regressors": rs, "their_public_lb": logged, **extra, "parity": parity},
                      models=[str(models / m) for m in cs + rs], gpu_hours=round(gpu_h, 2),
                      ms_per_query=round(1000 * dev_s / 2000, 1) if dev_s else None,
                      notes="ZhichunRoad's code, run as-is with adapters. Deviations: " + DEVIATIONS)
    if not a.no_write:
        (RUNS / "p0" / "zr_score_report.json").write_text(json.dumps(report, indent=1))
        import update_results
        update_results.main()


if __name__ == "__main__":
    main()
