"""Run logging (docs/0.md §0.1): every scored run writes runs/<phase>/<run>/metrics.json + per_query.parquet."""
import hashlib
import json
from functools import lru_cache

import pandas as pd

from mmsearch.esci import ESCI, RUNS
from mmsearch.metrics import by_slice, mean_ci, paired_bootstrap

BASELINE_RUN = ("p0", "zr-c123-r12")   # the ZR 5-member ensemble; every run is compared with it once it exists


@lru_cache(maxsize=1)
def data_version():
    h = hashlib.sha256((ESCI / "shopping_queries_dataset_examples.parquet").read_bytes()).hexdigest()
    return f"esci@{h[:16]}"


def load_per_query(phase, run):
    p = RUNS / phase / run / "per_query.parquet"
    return pd.read_parquet(p)["ndcg"] if p.exists() else None


def write_run(phase, run, per_query, splits, slices, *, args=None, models=None, gpu_hours=None,
              ms_per_query=None, report_test=False, notes=""):
    """per_query: Series query_id -> nDCG over all queries the run scored (dev and/or final test)."""
    out = RUNS / phase / run
    out.mkdir(parents=True, exist_ok=True)
    per_query.rename("ndcg").to_frame().to_parquet(out / "per_query.parquet")

    dev_q = splits.index[splits == "dev"]
    dev = per_query.reindex(dev_q)
    if dev.isna().any():
        raise ValueError(f"{run}: {dev.isna().sum()} dev queries unscored")
    ci = mean_ci(dev.to_numpy())
    rec = {"phase": phase, "run": run, "args": args or {}, "models": models or [], "data_version": data_version(),
           "dev": {"ndcg": ci["mean"], "ci95": ci["ci95"], "n": len(dev),
                   "slices": {c: by_slice(dev, slices.loc[dev_q, c]) for c in slices.columns}},
           "vs_baseline": None, "test": None, "gpu_hours": gpu_hours, "ms_per_query": ms_per_query, "notes": notes}

    base = load_per_query(*BASELINE_RUN)
    if base is not None and (phase, run) != BASELINE_RUN:
        rec["vs_baseline"] = paired_bootstrap(base.reindex(dev_q), dev)
    if report_test:
        test = per_query.reindex(splits.index[splits == "final_test"])
        if test.isna().any():
            raise ValueError(f"{run}: {test.isna().sum()} final-test queries unscored")
        rec["test"] = {"ndcg": float(test.mean()), "ci95": mean_ci(test.to_numpy())["ci95"], "n": len(test)}
    (out / "metrics.json").write_text(json.dumps(rec, indent=1))
    return rec
