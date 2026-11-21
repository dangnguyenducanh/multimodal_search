"""Phase 8 (docs/8.md) proxy evaluation on dev queries.
For each dev query with >= 1 Exact and >= 1 Substitute (or Complement) product: seed = its Exact product with the
smallest id; candidates = the query's other judged products; positives = its Substitutes (Complements).
Methods: random, text similarity to the seed (bi-encoder), image similarity (CLIP), reranker P(S) / P(C) with the
seed title as the query, P(S) + image similarity (weight fit by 5-fold CV over queries).
Reference (not item-to-item: it sees the query): ZR's P(S|query) from Phase 0.
  python scripts/p8_recs.py --run p4/<a run with a classification head>
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
from mmsearch.esci import RUNS, load_p1_products, load_pairs  # noqa: E402
from mmsearch.images import image_store  # noqa: E402
from mmsearch.metrics import micro_f1, mrr, paired_bootstrap, recall_at_k  # noqa: E402
from mmsearch.recs import image_sim, seed_rows  # noqa: E402

P8 = RUNS / "p8"


def task_rows(dev, pos_label):
    g = dev.groupby("query_id")
    ok = g["label"].apply(lambda s: (s == "E").any() and (s == pos_label).any())
    rows = []
    for q in ok[ok].index:
        d = g.get_group(q)
        seed = d.loc[d["label"] == "E", "product_id"].min()
        c = d[d["product_id"] != seed]
        rows.append(pd.DataFrame({"qid": q, "seed": seed, "product_id": c["product_id"], "label": c["label"]}))
    return pd.concat(rows, ignore_index=True)


def metrics(t, score, pos_label):
    out = {}
    for q, d in t.assign(s=score).groupby("qid"):
        ranked = d.sort_values(["s", "product_id"], ascending=[False, True])["product_id"].tolist()
        rel = d.loc[d["label"] == pos_label, "product_id"].tolist()
        out[q] = (mrr(ranked, rel), recall_at_k(ranked, rel, 5))
    df = pd.DataFrame(out, index=["mrr", "r5"]).T
    return df


def text_vectors():
    from mmsearch.retrieval import corpus
    name = next(n for n in ["bge-small-ft", "bge-small"] if (RUNS / "p7" / f"vec_{n}.npy").exists())
    v = np.load(RUNS / "p7" / f"vec_{name}.npy", mmap_mode="r")
    return name, pd.Series(np.arange(len(v)), index=corpus()["product_id"]), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="phase/name of a reranker run with a classification head")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from mmsearch.train import predict_run
    phase, name = a.run.split("/")
    P8.mkdir(parents=True, exist_ok=True)
    dev = load_pairs(splits_=("dev",))
    if a.limit:
        dev = dev[dev["query_id"].isin(dev["query_id"].drop_duplicates().head(a.limit))]
    products = load_p1_products().set_index("product_id")
    store = image_store()
    tname, pos, vec = text_vectors()
    report = {"reranker": a.run, "text_model": tname}
    for task, lab, pcol in [("alternatives", "S", "pS"), ("goes_with", "C", "pC")]:
        t = task_rows(dev, lab)
        rows = seed_rows(t["seed"].to_numpy(), t["product_id"].to_numpy(), products)
        sc = predict_run(phase, name, rows=rows)
        if sc[pcol].isna().all():
            sys.exit(f"{a.run} has no classification head")
        a_v, b_v = vec[pos.loc[t["seed"]].to_numpy()].astype(np.float32), vec[pos.loc[t["product_id"]].to_numpy()].astype(np.float32)
        txt = (a_v * b_v).sum(1)
        img = image_sim(t["seed"].to_numpy(), t["product_id"].to_numpy(), store)
        rng = np.random.default_rng(0)
        methods = {"random": None, "text_sim": txt, "image_sim": np.nan_to_num(img, nan=-1.0), f"reranker_{pcol}": sc[pcol].to_numpy()}
        res = {}
        rand = [metrics(t, rng.random(len(t)), lab) for _ in range(20)]
        res["random"] = sum(rand) / len(rand)
        for m, s in methods.items():
            if s is not None:
                res[m] = metrics(t, s, lab)
        # P + image similarity, weight by 5-fold CV over queries (out of fold)
        fused = np.zeros(len(t))
        zi = np.nan_to_num(img, nan=np.nanmin(img) if np.isfinite(img).any() else 0)
        for tr, te in GroupKFold(5).split(t, groups=t["qid"]):
            best = max(((metrics(t.iloc[tr], sc[pcol].to_numpy()[tr] + w * zi[tr], lab)["mrr"].mean(), w)
                        for w in np.linspace(0, 1, 11)))
            fused[te] = sc[pcol].to_numpy()[te] + best[1] * zi[te]
        res[f"reranker_{pcol}+image"] = metrics(t, fused, lab)
        # reference: ZR's P(S|query) over the same candidates (uses the query; not item-to-item)
        zr = RUNS / "p0" / "zr_pred" / "member_c1_dev.parquet"
        if zr.exists():
            zp = np.mean([pd.read_parquet(RUNS / "p0" / "zr_pred" / f"member_{m}_dev.parquet").set_index(["query_id", "product_id"])
                          .filter(like="s").to_numpy() for m in ["c1", "c2", "c3"]], axis=0)
            idx = pd.read_parquet(zr)[["query_id", "product_id"]]
            zcol = {"S": 3, "C": 0}[lab]                                       # their order C, E, I, S
            zs = pd.Series(zp[:, zcol], index=pd.MultiIndex.from_frame(idx)).reindex(pd.MultiIndex.from_arrays([t["qid"], t["product_id"]]))
            res["reference_zr_p(label|query)"] = metrics(t, zs.to_numpy(), lab)
        rep = {"queries": int(t["qid"].nunique()), "pairs": len(t)}
        best_base = max(["random", "text_sim", "image_sim"], key=lambda m: res[m]["mrr"].mean())
        for m, df in res.items():
            rep[m] = {"mrr": float(df["mrr"].mean()), "recall@5": float(df["r5"].mean())}
            if m not in ("random", "text_sim", "image_sim"):
                rep[m]["vs_best_baseline_mrr"] = paired_bootstrap(res[best_base]["mrr"], df["mrr"].reindex(res[best_base].index))
        report[task] = rep
        print(task, json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "vs_best_baseline_mrr"})
                                for k, v in rep.items()})[:600], flush=True)
    # Task 3 style: substitute vs not on all judged dev pairs, from the reranker's query-based probabilities
    full = predict_run(phase, name, "dev")
    pred = np.array(["E", "S", "C", "I"])[full[["pE", "pS", "pC", "pI"]].to_numpy().argmax(1)]
    report["task3_substitute_f1"] = micro_f1(pred == "S", full["label"].to_numpy() == "S")
    zr_rep = RUNS / "p0" / "zr_score_report.json"
    if zr_rep.exists():
        report["task3_substitute_f1_zr"] = json.loads(zr_rep.read_text()).get("zr-c123", {}).get("task3_substitute_f1")
    (P8 / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("task3 substitute F1", report["task3_substitute_f1"], "ZR", report.get("task3_substitute_f1_zr"))


if __name__ == "__main__":
    main()
