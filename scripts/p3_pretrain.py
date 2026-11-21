"""Phase 3 (docs/3.md): pretrain variants at proxy scale, then fine-tune "ZR recipe @ proxy" from each and compare
with the same recipe from vanilla weights (paired bootstrap on dev). A winner (CI > 0) is written to
runs/p3/best_backbone.txt, which the Phase 4 ladder's 'pretrained' step picks up.
  python scripts/p3_pretrain.py [--variants zr,cheap,real] [--products 200000]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mmsearch.esci import RUNS  # noqa: E402
from mmsearch.metrics import paired_bootstrap  # noqa: E402
from mmsearch.runlog import load_per_query  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="zr,cheap,real")
    ap.add_argument("--products", type=int, default=200_000)
    a = ap.parse_args()
    splits = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
    dev_q = splits.index[splits == "dev"]
    vanilla = load_per_query("p4", "zr-proxy")
    if vanilla is None:
        sys.exit("needs runs/p4/zr-proxy (the ladder's first run)")
    report = {}
    for v in a.variants.split(","):
        out = RUNS / "p3" / v
        if not (out / "pretrain_meta.json").exists():
            subprocess.run([PY, "-c", f"from mmsearch.pretrain import run; run('{v}', n_products={a.products})"],
                           cwd=ROOT, check=True)
        name = f"zr-proxy-from-{v}"
        if load_per_query("p3", name) is None:
            subprocess.run([PY, "-m", "mmsearch.train", json.dumps({"name": name, "phase": "p3", "backbone": str(out)})],
                           cwd=ROOT, check=True)
        r = paired_bootstrap(vanilla.reindex(dev_q), load_per_query("p3", name).reindex(dev_q))
        meta = json.loads((out / "pretrain_meta.json").read_text())
        report[v] = {"vs_vanilla": r, "pretrain_gpu_hours": meta["gpu_hours"], "held_out": meta["held_out_after"]}
        print(v, json.dumps(report[v]), flush=True)
    wins = {v: r for v, r in report.items() if r["vs_vanilla"]["ci95"][0] > 0}
    if wins:
        best = max(wins, key=lambda v: wins[v]["vs_vanilla"]["gain"])
        (RUNS / "p3" / "best_backbone.txt").write_text(str(RUNS / "p3" / best))
        report["winner"] = best
    else:
        report["winner"] = None
        report["decision"] = "dropped: no variant beat vanilla weights (CI above 0)"
    (RUNS / "p3" / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
