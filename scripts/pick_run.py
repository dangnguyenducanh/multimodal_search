"""Print a run for the project queue:  --best (the ladder's current best), --cls (best ladder run with a
classification head: needed for distillation, pseudo-labels and recommendations), --full (the 4.2 full-scale run).
Prints phase/name; prints nothing if no such run exists yet."""
import json
import sys
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main():
    log = json.loads((RUNS / "p4" / "ladder.json").read_text()) if (RUNS / "p4" / "ladder.json").exists() else {}
    if "--best" in sys.argv:
        if log.get("best"):
            print(f"p4/{log['best']}")
        return
    if "--full" in sys.argv:
        f = RUNS / "p4" / "full_run.txt"
        if f.exists():
            print(f.read_text().strip())
        return
    best = None
    for d in sorted((RUNS / "p4").glob("*/cfg.json")):
        cfg = json.loads(d.read_text())
        m = d.parent / "metrics.json"
        if cfg.get("head") in ("cls", "both") and m.exists() and not cfg.get("oof") and not cfg.get("max_rows"):
            nd = json.loads(m.read_text())["dev"]["ndcg"]
            if best is None or nd > best[0]:
                best = (nd, d.parent.name)
    if best:
        print(f"p4/{best[1]}")


if __name__ == "__main__":
    main()
