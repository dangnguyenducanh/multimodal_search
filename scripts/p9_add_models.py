"""Phase 9 demo tables (docs/9.md, docs/demo-handoff.md): put the trained models into runs/p9/catalog.sqlite.

Partial by design: each part runs only when its inputs exist and says what it skipped, so the queue can call it at
any point (scripts/run_all.sh, last step); rerunning refreshes everything it can.
  models       name, role, dev_ndcg, ci_low, ci_high, test_ndcg, built_at, ms_per_query (from each run's metrics.json)
  pair_scores  (query_id, product_id, model) -> score, pE, pS, pC, pI for every dev + final-test pair, per role:
               zr          ZR's 5 members (runs/p0/zr_pred) combined by mmsearch.ensemble.zr_score
               ours        our text winner: scripts/pick_run.py --full, else --best
               multimodal  the better of runs/p5/{caption-field,image-in-head}, when trained
  examples     10 gained + 5 lost + 5 random dev queries by per-query nDCG gain of ours over zr
  recs         per product in the examples' judged lists: BM25 top 50 by its title, scored by the
               scripts/pick_run.py --cls run; top 10 by P(S) = alternatives, by P(C) = goes_with
  python scripts/p9_add_models.py [--db PATH] [--zr-pred DIR] [--no-recs]
GPU (predict_run: a run's missing final-test scores, and recs) only when one is visible, capped at MMS_GPU_GB
(default 24 here; the project's GPU jobs together stay under 25 GB, see scripts/run_all.sh).
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"  # not mmsearch.esci.RUNS: importing mmsearch with MMS_GPU_GB set touches the GPU
ZR_RUN = "p0/zr-c123-r12" if (RUNS / "p0" / "zr-c123-r12" / "per_query.parquet").exists() else "p0/zr-c123-r1"
MM_RUNS = ["p5/caption-field", "p5/image-in-head"]
SPLITS = ["dev", "final_test"]
ZR_TO_OURS = [1, 3, 0, 2]  # their prob columns (C, E, I, S) -> ours (E, S, C, I), as scripts/p6_ensemble.py
SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_scores(query_id INTEGER, product_id TEXT, model TEXT, score REAL,
                                       pE REAL, pS REAL, pC REAL, pI REAL, PRIMARY KEY(query_id, model, product_id))
    WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS recs(product_id TEXT, kind TEXT, rank INTEGER, rec_product_id TEXT, score REAL,
                                PRIMARY KEY(product_id, kind, rank)) WITHOUT ROWID;
"""


def prepare(db):
    db.executescript(SCHEMA)
    if "ms_per_query" not in [r[1] for r in db.execute("PRAGMA table_info(models)")]:
        db.execute("ALTER TABLE models ADD COLUMN ms_per_query REAL")


def metrics(run):
    f = RUNS / run / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else {}


def write_model(db, name, role, pairs, m):
    """Replace the role's model: its models row and its pair_scores. pairs: query_id, product_id, score, pE..pI."""
    for (old,) in db.execute("SELECT name FROM models WHERE role = ? OR name = ?", (role, name)).fetchall():
        db.execute("DELETE FROM pair_scores WHERE model = ?", (old,))
    db.execute("DELETE FROM models WHERE role = ? OR name = ?", (role, name))
    dev, test = m.get("dev") or {}, m.get("test") or {}
    lo, hi = dev.get("ci95") or (None, None)
    db.execute("INSERT INTO models(name, role, dev_ndcg, ci_low, ci_high, test_ndcg, built_at, ms_per_query) "
               "VALUES (?,?,?,?,?,?,?,?)", (name, role, dev.get("ndcg"), lo, hi, test.get("ndcg"),
                                            time.strftime("%Y-%m-%d %H:%M"), m.get("ms_per_query")))
    vals = pairs[["score", "pE", "pS", "pC", "pI"]].astype(float).to_numpy().tolist()
    db.executemany("INSERT INTO pair_scores VALUES (?,?,?,?,?,?,?,?)",
                   [(int(q), str(p), name, *[None if v != v else v for v in vs])  # NaN -> NULL (no cls head)
                    for q, p, vs in zip(pairs["query_id"], pairs["product_id"], vals)])
    print(f"{role}: {name} ({len(pairs):,} pairs, dev nDCG {dev.get('ndcg')})")


def zr_pairs(pred):
    from mmsearch.ensemble import zr_score
    out = []
    for split in SPLITS:
        regs = ["r1", "r2"] if (pred / f"member_r2_{split}.parquet").exists() else ["r1"]   # r2 = stage 3, last
        files = {m: pred / f"member_{m}_{split}.parquet" for m in ["c1", "c2", "c3", *regs]}
        if not all(f.exists() for f in files.values()):
            print(f"zr: skip {split} (members missing in {pred})")
            continue
        d = {m: pd.read_parquet(f) for m, f in files.items()}
        base = d["c1"][["query_id", "product_id"]]
        assert all(o[["query_id", "product_id"]].equals(base) for o in d.values()), f"zr {split}: rows not aligned"
        cls = [d[m].filter(like="s").to_numpy() for m in ["c1", "c2", "c3"]]
        p = np.mean(np.stack([c.astype(np.float32) for c in cls]), axis=0)[:, ZR_TO_OURS]
        out.append(base.assign(score=zr_score(cls, [d[m].filter(like="s").to_numpy() for m in regs]),
                               pE=p[:, 0], pS=p[:, 1], pC=p[:, 2], pI=p[:, 3]))
    return pd.concat(out, ignore_index=True) if out else None


def run_pairs(run, gpu):
    from mmsearch.train import predict_run
    out = []
    for split in SPLITS:
        if not (RUNS / run / f"scores_{split}.parquet").exists() and not (gpu and (RUNS / run / "model.pt").exists()):
            print(f"{run}: skip {split} (no scores_{split}.parquet, and no GPU + model.pt to predict it)")
            continue
        out.append(predict_run(*run.split("/"), split)[["query_id", "product_id", "score", "pE", "pS", "pC", "pI"]])
    return pd.concat(out, ignore_index=True) if out else None


def pick(flag):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "pick_run.py"), flag],
                          capture_output=True, text=True, check=True).stdout.strip()


def write_examples(db, gain, seed=0):
    """gain: Series dev query_id -> nDCG(ours) - nDCG(zr). Candidates as scripts/p9_build_db.py: >= 10 judged
    products, >= 90% with an image, >= 2 words."""
    st = db.execute("""SELECT j.query_id, q.query FROM judgments j JOIN products p ON p.product_id = j.product_id
                       JOIN queries q ON q.query_id = j.query_id GROUP BY j.query_id
                       HAVING count(*) >= 10 AND avg(p.has_image) >= 0.9""").fetchall()
    g = gain[gain.index.isin([q for q, text in st if len(text.split()) >= 2])].sort_index().sort_values(kind="stable")
    gained, lost = list(g[g > 0].index[::-1][:10]), list(g[g < 0].index[:5])
    rest = g.index.difference(gained + lost)
    rand = list(np.random.default_rng(seed).choice(rest, min(5, len(rest)), replace=False))
    picks = [(q, "gained") for q in gained] + [(q, "lost") for q in lost] + [(q, "random") for q in rand]
    db.execute("DELETE FROM examples")
    db.executemany("INSERT INTO examples VALUES (?,?,?)", [(i + 1, int(q), r) for i, (q, r) in enumerate(picks)])
    print(f"examples: {len(gained)} gained, {len(lost)} lost, {len(rand)} random")


def write_recs(db, run, k=10, n=50):
    from app.services.pipeline import retrieve
    from mmsearch.esci import load_p1_products
    from mmsearch.recs import dedupe, seed_rows
    from mmsearch.train import predict_run
    products = load_p1_products().set_index("product_id")
    seeds, cands = [], []
    for (s,) in db.execute("SELECT DISTINCT j.product_id FROM examples e JOIN judgments j USING (query_id) "
                           "ORDER BY 1").fetchall():
        title = products.at[s, "product_title"] if s in products.index else None
        hits = [h for h in retrieve(db, title or "", n + 1) if h.product_id != s][:n]
        c = [h.product_id for h, ok in zip(hits, dedupe(title, [h.title for h in hits])) if ok]
        seeds += [s] * len(c)
        cands += c
    if not cands:
        print("recs: skipped (no candidates)")
        return
    rows = seed_rows(np.array(seeds), np.array(cands), products)
    sc = predict_run(*run.split("/"), rows=rows)
    if sc["pS"].isna().all():
        print(f"recs: skipped ({run} has no classification head)")
        return
    rows = rows.assign(pS=sc["pS"].to_numpy(), pC=sc["pC"].to_numpy())
    db.execute("DELETE FROM recs")
    for kind, col in [("alternatives", "pS"), ("goes_with", "pC")]:
        top = rows.sort_values(["seed", col, "product_id"], ascending=[True, False, True]).groupby("seed").head(k)
        rank = top.groupby("seed").cumcount() + 1
        db.executemany("INSERT INTO recs VALUES (?,?,?,?,?)", [(s, kind, int(r), p, float(v)) for s, r, p, v in
                                                               zip(top["seed"], rank, top["product_id"], top[col])])
    print(f"recs: {len(set(seeds)):,} products, {len(rows):,} pairs scored by {run}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(RUNS / "p9" / "catalog.sqlite"))
    ap.add_argument("--zr-pred", default=str(RUNS / "p0" / "zr_pred"))
    ap.add_argument("--no-recs", action="store_true", help="skip recs (default: run when pick_run.py --cls names a run)")
    a = ap.parse_args()
    os.environ.setdefault("MMS_GPU_GB", "24")  # read by mmsearch/__init__.py on first import
    import torch
    gpu = torch.cuda.is_available()
    db = sqlite3.connect(a.db)
    db.row_factory = sqlite3.Row  # app.services.pipeline.retrieve reads columns by name
    prepare(db)

    zr = zr_pairs(Path(a.zr_pred))
    if zr is not None:
        write_model(db, ZR_RUN, "zr", zr, metrics(ZR_RUN))
    ours = next((r for r in (pick("--full"), pick("--best")) if r and (RUNS / r / "scores_dev.parquet").exists()), None)
    mm = max((r for r in MM_RUNS if (RUNS / r / "scores_dev.parquet").exists() and metrics(r)),
             key=lambda r: metrics(r)["dev"]["ndcg"], default=None)
    for role, run in [("ours", ours), ("multimodal", mm)]:
        pairs = run_pairs(run, gpu) if run else None
        if pairs is None:
            print(f"{role}: skipped (no trained run with scores yet)")
        else:
            write_model(db, run, role, pairs, metrics(run))

    pq = {r: RUNS / r / "per_query.parquet" for r in (ZR_RUN, ours) if r}
    if ours and len(pq) == 2 and all(f.exists() for f in pq.values()):
        sp = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
        dev = sp.index[sp == "dev"]
        z, o = (pd.read_parquet(pq[r])["ndcg"].reindex(dev) for r in (ZR_RUN, ours))
        write_examples(db, (o - z).dropna())
    else:
        print("examples: skipped (needs per_query.parquet of zr and ours)")

    cls = pick("--cls")
    if a.no_recs or not cls or not gpu:
        print(f"recs: skipped ({'--no-recs' if a.no_recs else 'no --cls run yet' if not cls else 'no GPU visible'})")
    else:
        write_recs(db, cls)
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"{a.db} {os.path.getsize(a.db) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
