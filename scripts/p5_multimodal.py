"""Phase 5 (docs/5.md): does the product photo improve the text reranker?
  python scripts/p5_multimodal.py --variants A      # late fusion over a text run's scores (CPU)
  python scripts/p5_multimodal.py --variants B,C    # caption field / image embedding in the head (GPU, proxy scale)
The text run defaults to the Phase 4 ladder's current best (runs/p4/ladder.json)."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmsearch.esci import RUNS, load_pairs  # noqa: E402
from mmsearch.fusion import feature_table, fit_predict, oof  # noqa: E402
from mmsearch.metrics import by_slice, ndcg_per_query, paired_bootstrap  # noqa: E402
from mmsearch.runlog import load_per_query, write_run  # noqa: E402


def ladder_best():
    log = json.loads((RUNS / "p4" / "ladder.json").read_text())
    return log.get("best", "zr-proxy"), log.get("best_over", {})


def splits_slices():
    return (pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"], pd.read_parquet(RUNS / "p0" / "slices.parquet"))


def variant_a(text_run, report):
    from mmsearch.train import predict_run
    splits, slices = splits_slices()
    queries = load_pairs(splits_=("dev", "final_test")).drop_duplicates("query_id").set_index("query_id")["query"]
    dev = feature_table(predict_run("p4", text_run, "dev"), queries)
    base = ndcg_per_query(dev)
    dev_q = splits.index[splits == "dev"]
    prev, results = None, {}
    for kind in ["A0", "A1", "A2", "A3"]:
        if kind in ("A2", "A3") and not results.get("A1", {}).get("beats_A0"):
            results[kind] = {"skipped": "A1 did not beat A0 (docs/5.md rule)"}
            continue
        s, info = oof(dev, kind)
        pq = ndcg_per_query(s).reindex(dev_q)
        vs_text = paired_bootstrap(base.reindex(dev_q), pq)
        r = {"oof_dev_ndcg": float(pq.mean()), "vs_text": vs_text, **info,
             "slices_gain": {c: by_slice(pq - base.reindex(dev_q), slices.loc[dev_q, c]) for c in ["visual", "image"]}}
        if kind == "A1":
            r["beats_A0"] = paired_bootstrap(prev, pq)["ci95"][0] > 0
        results[kind] = r
        prev = pq
        write_run("p5", f"fusion-{kind}", pq, splits, slices, args={"text_run": text_run, "combiner": kind},
                  notes=f"late fusion {kind} over {text_run}; dev number is out-of-fold (GroupKFold 5)")
        print(kind, json.dumps({k: v for k, v in r.items() if k != "importance"}, default=str)[:400], flush=True)
    ok = {k: v for k, v in results.items() if "oof_dev_ndcg" in v and v["vs_text"]["ci95"][0] > 0}
    if ok:                                                      # winner -> refit on dev, final test once
        win = max(ok, key=lambda k: ok[k]["oof_dev_ndcg"])
        test = feature_table(predict_run("p4", text_run, "final_test"), queries)
        s, _ = fit_predict(win, dev, test)
        pq = pd.concat([prev_dev_scores(win), ndcg_per_query(test.assign(score=s))])
        write_run("p5", f"fusion-{win}", pq, splits, slices, report_test=True, args={"text_run": text_run, "combiner": win},
                  notes=f"late fusion {win} over {text_run}; dev out-of-fold, final test from a refit on all of dev")
        results["winner"] = win
    report["A"] = results


def prev_dev_scores(kind):
    return load_per_query("p5", f"fusion-{kind}").dropna()


def variant_trained(name, extra, text_run, best_over, report):
    py = str(ROOT / ".venv" / "bin" / "python")
    cfg = {**best_over, **extra, "name": name, "phase": "p5"}
    if load_per_query("p5", name) is None:
        subprocess.run([py, "-m", "mmsearch.train", json.dumps(cfg)], cwd=ROOT, check=True)
    splits, slices = splits_slices()
    dev_q = splits.index[splits == "dev"]
    a, b = load_per_query("p4", text_run).reindex(dev_q), load_per_query("p5", name).reindex(dev_q)
    report[name] = {"vs_text": paired_bootstrap(a, b),
                    "slices_gain": {c: by_slice(b - a, slices.loc[dev_q, c]) for c in ["visual", "image"]}}
    print(name, json.dumps(report[name], default=str)[:400], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A")
    ap.add_argument("--text-run", default=None)
    a = ap.parse_args()
    text_run, best_over = ladder_best()
    text_run = a.text_run or text_run
    path = RUNS / "p5" / "report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report["text_run"] = text_run
    v = a.variants.split(",")
    if "A" in v:
        variant_a(text_run, report)
    if "B" in v:
        variant_trained("caption-field", {"order": "caption"}, text_run, best_over, report)
    if "C" in v:
        variant_trained("image-in-head", {"image": True}, text_run, best_over, report)
    path.write_text(json.dumps(report, indent=1, default=str))
    import update_results  # noqa: E402
    update_results.main()


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    main()
