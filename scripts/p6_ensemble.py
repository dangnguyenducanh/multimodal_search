"""Phase 6 (docs/6.md): from the ZR 5-member ensemble, one change at a time, each kept only if it beats the stage
before on dev (paired bootstrap, CI > 0). Fitted combiners are scored out of fold (5-fold GroupKFold over dev
queries). Inputs are saved member outputs only (no GPU):
  ZR members  runs/p0/zr_pred/member_{c1,c2,c3,r1,r2}_{dev,final_test}.parquet
  our models  runs/p4|p5/<run>/scores_{dev,final_test}.parquet  (listed with --ours / --images / --bags)
  python scripts/p6_ensemble.py --ours p4/<best>,p4/<full> --images p5/caption-field --bags p4/<seed runs>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from mmsearch.ensemble import expected_gain, fit_weights, greedy_prune, rank_norm, weighted, zr_score  # noqa: E402
from mmsearch.esci import RUNS, load_pairs  # noqa: E402
from mmsearch.metrics import ndcg_per_query, paired_bootstrap  # noqa: E402
from mmsearch.runlog import load_per_query, write_run  # noqa: E402

ZR_TO_OURS = [1, 3, 0, 2]      # their prob columns (C, E, I, S) -> ours (E, S, C, I)


def member_tables():
    """Per split: labels + one score column per member (rank-normalised) + raw columns for the ZR formula."""
    out = {}
    for split in ["dev", "final_test"]:
        lab = load_pairs(splits_=(split,))[["query_id", "product_id", "label"]]
        t = lab.copy()
        raw = {}
        for m in ["c1", "c2", "c3", "r1", "r2"]:
            f = RUNS / "p0" / "zr_pred" / f"member_{m}_{split}.parquet"
            if not f.exists():
                continue
            d = t[["query_id", "product_id"]].merge(pd.read_parquet(f), on=["query_id", "product_id"], how="left")
            v = d.filter(like="s").to_numpy()
            raw[m] = v
            t[f"zr_{m}"] = rank_norm(expected_gain(v[:, ZR_TO_OURS]) if m[0] == "c" else v[:, 0], t["query_id"])
        out[split] = (t, raw)
    return out


def add_ours(tables, runs):
    for r in runs:
        phase, name = r.split("/")
        for split, (t, _) in tables.items():
            f = RUNS / phase / name / f"scores_{split}.parquet"
            if not f.exists():
                from mmsearch.train import predict_run
                predict_run(phase, name, split)
            s = t[["query_id", "product_id"]].merge(pd.read_parquet(f)[["query_id", "product_id", "score"]],
                                                   on=["query_id", "product_id"], how="left")["score"]
            t[f"{phase}:{name}"] = rank_norm(s.to_numpy(), t["query_id"])


def nd(t, score):
    return ndcg_per_query(t.assign(score=score))


def cv_weights(t, members, n=5):
    """Out-of-fold scores of coordinate-ascent weights (fit on 4/5 of dev queries, score the 5th)."""
    out = np.zeros(len(t))
    for a, b in GroupKFold(n_splits=n).split(t, groups=t["query_id"]):
        tr = t.iloc[a]
        w, _ = fit_weights(tr, members, lambda s: nd(tr, s).mean())
        out[b] = weighted(t.iloc[b], members, w)
    w_all, _ = fit_weights(t, members, lambda s: nd(t, s).mean())
    return out, w_all


def cv_stack(t, members, n=5):
    import lightgbm as lgb
    lab = t["label"].map({"I": 0, "C": 1, "S": 2, "E": 3})
    out = np.zeros(len(t))
    for a, b in GroupKFold(n_splits=n).split(t, groups=t["query_id"]):
        tr = t.iloc[a].sort_values("query_id")
        m = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1, 10, 100], num_leaves=15, min_data_in_leaf=50,
                           learning_rate=0.05, n_estimators=200, verbose=-1)
        # numpy, not the DataFrame: LightGBM rejects feature names with ':' or ',' (our run names have both)
        m.fit(tr[members].to_numpy(), lab.loc[tr.index], group=tr.groupby("query_id", sort=False).size().to_numpy())
        out[b] = m.predict(t.iloc[b][members].to_numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="")
    ap.add_argument("--images", default="")
    ap.add_argument("--bags", default="")
    a = ap.parse_args()
    split_s = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
    slices = pd.read_parquet(RUNS / "p0" / "slices.parquet")
    tables = member_tables()
    t, raw = tables["dev"]
    if len(raw) < 5:
        sys.exit(f"Phase 0 ZR members missing ({sorted(raw)}); run after the Phase 0 queue finishes")
    ours = [r for r in a.ours.split(",") if r]
    images = [r for r in a.images.split(",") if r]
    bags = [r for r in a.bags.split(",") if r]
    add_ours(tables, ours + images + bags)
    t, raw = tables["dev"]
    zr_cols = [f"zr_{m}" for m in ["c1", "c2", "c3", "r1", "r2"]]
    col = lambda rs: [f"{r.split('/')[0]}:{r.split('/')[1]}" for r in rs]       # noqa: E731

    stages = []
    s0 = zr_score([raw[m] for m in ["c1", "c2", "c3"]], [raw[m] for m in ["r1", "r2"]])
    base_pq = load_per_query("p0", "zr-c123-r12")
    pq0 = nd(t, s0)
    assert np.allclose(pq0.sort_index().to_numpy(), base_pq.reindex(pq0.index).sort_index().to_numpy(), atol=1e-9), \
        "stage 0 does not reproduce the Phase 0 baseline"
    stages.append(("stage0-zr", zr_cols, pq0, "ZR sort_key_4 over the 5 Phase 0 members (= the baseline)"))
    c_eg = expected_gain(np.mean([raw[m][:, ZR_TO_OURS] for m in ["c1", "c2", "c3"]], axis=0))
    s1 = rank_norm(c_eg, t["query_id"]) + rank_norm(np.mean([raw["r1"][:, 0], raw["r2"][:, 0]], axis=0), t["query_id"])
    stages.append(("stage1-expected-gain", zr_cols, nd(t, s1), "expected gain + mean regression, rank-normalised"))
    cands = [("stage2-fitted-weights", zr_cols, "coordinate-ascent weights over the 5 ZR members"),
             ("stage3-family-diversity", zr_cols + col(ours), "+ our Phase 4 models"),
             ("stage4-bagging", zr_cols + col(ours) + col(bags), "+ seed/fold bags of our best member"),
             ("stage5-image-members", zr_cols + col(ours) + col(bags) + col(images), "+ Phase 5 image members")]
    for name, members, note in cands:
        if members == stages[-1][1] and name != "stage2-fitted-weights":
            stages.append((name, members, None, note + " (n/a: no new members given)"))
            continue
        s, _ = cv_weights(t, members)
        stages.append((name, members, nd(t, s), note))
    report, best_i = {"stages": []}, 0
    for i, (name, members, pq, note) in enumerate(stages):
        rec = {"stage": name, "members": members, "note": note}
        if pq is not None:
            rec["dev_ndcg"] = float(pq.mean())
            if i:
                r = paired_bootstrap(stages[best_i][2], pq)
                rec["vs_previous_best"] = r
                if r["ci95"][0] > 0:
                    best_i = i
            write_run("p6", name, pq, split_s, slices, args={"members": members}, notes=note)
        report["stages"].append(rec)
    win_members = stages[best_i][1]
    stack = cv_stack(t, win_members)
    pq6 = nd(t, stack)
    r6 = paired_bootstrap(stages[best_i][2], pq6)
    write_run("p6", "stage6-stacking", pq6, split_s, slices, args={"members": win_members},
              notes="LightGBM LambdaRank over member scores, out of fold")
    report["stages"].append({"stage": "stage6-stacking", "dev_ndcg": float(pq6.mean()), "vs_previous_best": r6})
    stacked = r6["ci95"][0] > 0

    cost = {}
    for m in win_members:
        if m.startswith("zr_"):
            tm = json.loads((RUNS / "p0" / "zr_pred" / "timing.json").read_text())
            cost[m] = 1000 * tm.get(f"{m[3:]}_dev_s", 0) / 2000
        else:
            ph, nm = m.split(":")
            cost[m] = json.loads((RUNS / ph / nm / "metrics.json").read_text()).get("ms_per_query") or 0
    w_all = cv_weights(t, win_members)[1]
    keep, _ = greedy_prune(win_members, lambda ms: nd(t, weighted(t, ms, {k: w_all[k] for k in ms})).mean(), cost,
                           tol=0.0005)
    s7, w7 = cv_weights(t, keep)
    pq7 = nd(t, s7)
    write_run("p6", "stage7-pruned", pq7, split_s, slices, args={"members": keep},
              ms_per_query=round(sum(cost[m] for m in keep), 1), notes="greedy backward pruning (tolerance 0.0005)")
    report["prune"] = {"kept": keep, "dropped": [m for m in win_members if m not in keep], "dev_ndcg": float(pq7.mean()),
                       "ms_per_query": sum(cost[m] for m in keep), "ms_full": sum(cost.values())}
    report["leave_one_out"] = {m: float(pq7.mean() - nd(t, weighted(t, [x for x in keep if x != m], w7)).mean())
                               for m in keep if len(keep) > 1}
    report["correlation"] = t[win_members].corr().round(3).to_dict()

    # final test once: the winning composition, weights refit on all of dev
    tt, _ = tables["final_test"]
    final = "stage6-stacking" if stacked else ("stage7-pruned" if pq7.mean() >= stages[best_i][2].mean() - 0.0005
                                               else stages[best_i][0])
    members = keep if final == "stage7-pruned" else win_members
    if final == "stage6-stacking":
        import lightgbm as lgb
        lab = t["label"].map({"I": 0, "C": 1, "S": 2, "E": 3})
        tr = t.sort_values("query_id")
        m = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1, 10, 100], num_leaves=15, min_data_in_leaf=50,
                           learning_rate=0.05, n_estimators=200, verbose=-1)
        m.fit(tr[members].to_numpy(), lab.loc[tr.index], group=tr.groupby("query_id", sort=False).size().to_numpy())
        st = m.predict(tt[members].to_numpy())
    elif final in ("stage0-zr", "stage1-expected-gain"):       # fixed formulas: apply them as they are
        _, rt = tables["final_test"]
        if final == "stage0-zr":
            st = zr_score([rt[m] for m in ["c1", "c2", "c3"]], [rt[m] for m in ["r1", "r2"]])
        else:
            ceg = expected_gain(np.mean([rt[m][:, ZR_TO_OURS] for m in ["c1", "c2", "c3"]], axis=0))
            st = rank_norm(ceg, tt["query_id"]) + rank_norm(np.mean([rt["r1"][:, 0], rt["r2"][:, 0]], axis=0), tt["query_id"])
    else:
        w = cv_weights(t, members)[1]
        st = weighted(tt, members, w)
    dev_pq = load_per_query("p6", final).dropna()
    write_run("p6", final, pd.concat([dev_pq, nd(tt, st)]), split_s, slices, report_test=True,
              args={"members": members}, notes=f"Phase 6 winner ({final}); final test scored once")
    report["winner"] = final
    (RUNS / "p6" / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "correlation"}, indent=1, default=str)[:3000])
    import update_results
    update_results.main()


if __name__ == "__main__":
    main()
