"""Phase 4.2 (docs/4.md): the ladder's winning recipe on the full US train pool with the large backbone,
1 epoch, scored on dev and final test (once). Compared with ZR r1 (single vs single) and the ZR ensemble."""
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

LARGE = {"FacebookAI/xlm-roberta-base": "FacebookAI/xlm-roberta-large",
         "microsoft/deberta-v3-base": "microsoft/deberta-v3-large"}


def main():
    log = json.loads((RUNS / "p4" / "ladder.json").read_text())
    over = dict(log.get("best_over", {}))
    backbone = over.get("backbone", "FacebookAI/xlm-roberta-base")
    over["backbone"] = LARGE.get(backbone, backbone)            # a Phase 3 checkpoint stays as it is
    over["data"] = "us_esjp" if str(over.get("data", "")).endswith("esjp") else "us"
    over.update({"batch": 8, "accum": 32, "eval_splits": ["dev", "final_test"], "eval_every": 0.25})
    name = "full-" + log.get("best", "zr-proxy")
    if load_per_query("p4", name) is None:
        subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "mmsearch.train", json.dumps({"name": name, **over})],
                       cwd=ROOT, check=True)
    (RUNS / "p4" / "full_run.txt").write_text(f"p4/{name}")
    splits = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
    dev_q = splits.index[splits == "dev"]
    ours = load_per_query("p4", name).reindex(dev_q)
    rep = {"run": name, "cfg": over}
    for ref in ["zr-r1", "zr-c123-r12"]:
        b = load_per_query("p0", ref)
        if b is not None:
            rep[f"vs_{ref}"] = paired_bootstrap(b.reindex(dev_q), ours)
    (RUNS / "p4" / "full_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
