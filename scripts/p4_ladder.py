"""Phases 1/2/4 at proxy scale (docs/4.md §4.0–4.2, docs/2.md, docs/1.md proposal 5): one change at a time from
"ZR recipe @ proxy", each kept only if it beats the current best on dev (paired bootstrap, CI > 0), or for cost
cuts only if it doesn't lose (CI lower bound > -0.002). Resumable: finished runs (metrics.json) are reused.
Decisions -> runs/p4/ladder.json.   nohup .venv/bin/python scripts/p4_ladder.py > runs/p4/ladder.log 2>&1 &
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmsearch.ensemble import zr_score  # noqa: E402
from mmsearch.esci import RUNS, load_examples, task1_us_test  # noqa: E402
from mmsearch.metrics import ndcg_per_query, paired_bootstrap  # noqa: E402
from mmsearch.runlog import load_per_query, write_run  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")
ZPY = str(Path.home() / "miniconda3" / "envs" / "zr" / "bin" / "python")
LOG = RUNS / "p4" / "ladder.json"
NONINF = -0.002

STEPS = [  # (step, doc reference, rule, candidate overrides)
    ("speed", "4.md 4.1 #0, 2.md #3", "noninferior", [{"dynamic_pad": True, "precision": "bf16"}]),
    ("length", "2.md #4", "best", [{"max_len": 128}, {"max_len": 384}]),
    ("text", "2.md #5-7", "best", [{"fmt": "A"}, {"budget": {"bullets": 96, "desc": 96}}, {"clean": "www"}]),
    ("head", "4.md 4.1 #1", "best", [{"head": "cls", "score": "zr"}, {"head": "cls", "score": "eg"},
                                     {"head": "both", "score": "sum"}]),
    ("lr", "4.md 4.1 (hyperparameters)", "best", [{"lr": 2e-5}, {"lr": 3e-5}]),
    ("listwise", "4.md 4.1 #2", "best", [{"listwise": 0.5}]),
    ("tricks", "4.md 4.1 #3", "cheaper", [{"fgm": 0.0}, {"rdrop": 0.0}, {"ema": 0.0}]),
    ("awp", "4.md 4.1 #4", "best", [{"fgm": 0.0, "awp": 1e-4}]),
    ("smoothing", "4.md 4.1 #5", "best", [{"label_smoothing": 0.1}]),
    ("rebalance", "2.md #9", "best", [{"rebalance": "zr"}, {"rebalance": "zr_title"}]),
    ("markers", "2.md #11", "best", [{"brand_markers": True}]),
    ("pooling", "4.md 4.1 #8", "best", [{"msd": True, "wpool": True}]),
    ("side", "4.md 4.1 #9", "cheaper", [{"side": False}]),
    ("bag", "4.md 4.1 #9", "best", [{"bag": True}]),
    ("pool", "1.md #5", "best", [{"data": "task1_us_esjp"}]),
    ("backbone", "4.md 4.1 #10", "best", [{"backbone": "microsoft/deberta-v3-base"}]),
    ("distill", "4.md 4.1 #6, 2.md #10", "oof", None),
    ("pseudo", "4.md 4.1 #7", "pseudo", None),
    ("pretrained", "4.md 4.1 #11", "pretrained", None),
]


def name_of(over):
    if not over:
        return "zr-proxy"
    parts = []
    for k, v in sorted(over.items()):
        v = "-".join(f"{a}{b}" for a, b in v.items()) if isinstance(v, dict) else str(v).split("/")[-1]
        parts.append(f"{k}={v}")
    return "px-" + ",".join(parts).replace(" ", "")


def train(over, extra=None):
    cfg = {**over, **(extra or {})}
    name = cfg.pop("name", None) or name_of(over)
    if load_per_query("p4", name) is None:
        print(f"== train {name}: {json.dumps(cfg)}", flush=True)
        subprocess.run([PY, "-m", "mmsearch.train", json.dumps({"name": name, **cfg})], cwd=ROOT, check=True)
    return name


def dev_scores(name):
    s = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
    return load_per_query("p4", name).reindex(s.index[s == "dev"])


def compare(best, cand):
    return paired_bootstrap(dev_scores(best), dev_scores(cand))


def parity():
    """4.0: their trainer at proxy scale vs ours, same rows and seed (regression member, ZR recipe)."""
    ours = train({})
    theirs = "zr-proxy-theirs"
    if load_per_query("p4", theirs) is None:
        d = RUNS / "p4" / "zr_proxy"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "train.csv").exists():
            subprocess.run([PY, "-c", (
                "import sys; sys.path[:0]=['scripts','.']\n"
                "import pandas as pd\nfrom p0_zr_data import their_format, product_table, COLS\n"
                "from mmsearch.esci import load_pairs\n"
                "p=load_pairs(splits_=('train','dev')); prod=product_table()\n"
                "t=p[(p.split=='train')&(p.small_version==1)]; v=p[p.split=='dev']\n"
                "conv=lambda x: x.rename(columns={'locale':'product_locale','label':'esci_label'})[['query_id','query','product_id','product_locale','esci_label']]\n"
                f"their_format(conv(t), prod).sample(frac=1, random_state=12345).to_csv('{d}/train.csv', index=False)\n"
                f"their_format(conv(v), prod).to_csv('{d}/valid.csv', index=False)\n")], cwd=ROOT, check=True)
        hf = RUNS / "p0" / "hf" / "xlm-roberta-base"
        if not (hf / "pytorch_model.bin").exists():
            subprocess.run([PY, "-c", "from huggingface_hub import snapshot_download as s; s('FacebookAI/xlm-roberta-base', "
                            f"local_dir='{hf}', allow_patterns=['config.json','pytorch_model.bin','sentencepiece.bpe.model',"
                            "'tokenizer.json','tokenizer_config.json'])"], check=True)
        env = {**os.environ, "MODEL": str(hf), "TRAIN": str(d / "train.csv"), "VALID": str(d / "valid.csv"),
               "NAME": "proxy_r1", "MIXED_PRECISION": "fp16"}
        if not (RUNS / "p0" / "zr_models" / "proxy_r1" / "train_time.json").exists():
            subprocess.run(["bash", "scripts/p0_zr_train.sh", "r1"], cwd=ROOT, env=env, check=True)
        out = RUNS / "p4" / theirs
        subprocess.run([ZPY, "scripts/p0_zr_infer.py", "--eval", "--members", "r1", "--alias", "r1=proxy_r1",
                        "--out", str(out), "--splits", "dev"], cwd=ROOT, check=True,
                       env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "12345"})
        m = pd.read_parquet(out / "member_r1_dev.parquet")
        ex = load_examples(["query_id", "product_id", "product_locale", "esci_label", "small_version", "split"])
        lab = task1_us_test(ex)[["query_id", "product_id", "esci_label"]].rename(columns={"esci_label": "label"})
        df = m.merge(lab, on=["query_id", "product_id"])
        pq = ndcg_per_query(df.assign(score=zr_score([], [m["s0"].to_numpy()])))
        splits = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
        slices = pd.read_parquet(RUNS / "p0" / "slices.parquet")
        gpu_h = json.loads((RUNS / "p0" / "zr_models" / "proxy_r1" / "train_time.json").read_text())["seconds"] / 3600
        write_run("p4", theirs, pq, splits, slices, gpu_hours=round(gpu_h, 3),
                  notes="ZR's own regression trainer at proxy scale (parity control for our reimplementation)")
    r = compare(theirs, ours)
    ok = r["ci95"][0] <= 0 <= r["ci95"][1] and abs(r["gain"]) <= 0.003
    return {"ours": ours, "theirs": theirs, "ours_minus_theirs": r, "pass": ok}


def oof_runs(best_over):
    """3-fold out-of-fold class probabilities with the current recipe (head forced to 'both' for probabilities)."""
    over = {**best_over, "head": "both" if best_over.get("head", "reg") == "reg" else best_over["head"]}
    parts = []
    for k in range(3):
        n = train(over, {"name": f"{name_of(best_over)}-oof{k}", "oof": [3, k], "eval_splits": []})
        parts.append(pd.read_parquet(RUNS / "p4" / n / "scores_oof.parquet"))
    oof = pd.concat(parts, ignore_index=True)
    path = RUNS / "p4" / "oof_probs.parquet"
    oof.to_parquet(path, index=False)
    from mmsearch.augment import confident_filter
    lab = oof["label"].map({"E": 0, "S": 1, "C": 2, "I": 3}).to_numpy()
    drop = confident_filter(lab, oof[["pE", "pS", "pC", "pI"]].to_numpy())
    oof.loc[drop, ["query_id", "product_id"]].to_parquet(RUNS / "p4" / "cf_drop.parquet", index=False)
    return [{"distill": str(path)}, {"filter_drop": str(RUNS / "p4" / "cf_drop.parquet")}]


def main():
    log = json.loads(LOG.read_text()) if LOG.exists() else {"steps": []}
    done = {s["step"] for s in log["steps"]}
    if "parity" not in log:
        log["parity"] = parity()
        LOG.write_text(json.dumps(log, indent=1))
    best_over = log.get("best_over", {})
    best = name_of(best_over)
    for step, ref, rule, cands in STEPS:
        if step in done:
            continue
        if rule == "oof":
            cands = oof_runs(best_over)
            rule = "best"
        elif rule == "pseudo":
            p = RUNS / "p7" / "pseudo_labels.parquet"
            if not p.exists():
                print(f"== {step}: skipped for now (needs {p}, built after Phase 7)", flush=True)
                continue
            cands, rule = [{"pseudo": str(p)}], "best"
        elif rule == "pretrained":
            p = RUNS / "p3" / "best_backbone.txt"
            if not p.exists():
                print(f"== {step}: skipped for now (needs Phase 3)", flush=True)
                continue
            cands, rule = [{"backbone": p.read_text().strip()}], "best"
        if step == "smoothing" and best_over.get("head", "reg") == "reg":
            log["steps"].append({"step": step, "ref": ref, "decision": "n/a (no classification head)"})
            continue
        rec = {"step": step, "ref": ref, "rule": rule, "base": best, "candidates": []}
        winner = None
        for c in cands:
            over = {**best_over, **c}
            n = train(over)
            r = compare(best, n)
            rec["candidates"].append({"run": n, "change": c, **r})
            print(f"   {step}: {n} vs {best}: {r['gain']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]", flush=True)
            if rule == "best" and r["ci95"][0] > 0 and (winner is None or r["gain"] > winner[1]):
                winner = (over, r["gain"])
            if rule in ("noninferior", "cheaper") and r["ci95"][0] > NONINF:
                best_over, best = over, n                       # cost cut kept; later candidates build on it
                rec.setdefault("kept", []).append(c)
        if winner:
            best_over, best = winner[0], name_of(winner[0])
            rec["kept"] = [{k: v for k, v in winner[0].items() if log.get("best_over", {}).get(k) != v}]
        rec["decision"] = f"kept {rec.get('kept')}" if rec.get("kept") else "no change"
        log["steps"].append(rec)
        log["best_over"], log["best"] = best_over, best
        LOG.write_text(json.dumps(log, indent=1, default=str))
        print(f"== {step}: {rec['decision']} -> best {best}", flush=True)
    LOG.write_text(json.dumps(log, indent=1, default=str))
    print("ladder done; best", best, flush=True)


if __name__ == "__main__":
    main()
