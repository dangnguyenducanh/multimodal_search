"""Phase 0 §0.5: run ZhichunRoad's own predictor (run_origin.py, Task1Predictor) on our checkpoints. `zr` env.

  --eval  dev + final test. Each member's outputs come from THEIR predict_classification / predict_regression over
          THEIR dataloaders (incl. the c2/c3 field orders) -> runs/p0/zr_pred/member_<m>_<split>.parquet (cached per
          member, so partial ensembles can be scored early). When all 5 exist, THEIR predict() runs end to end with
          those outputs and writes its ranking file -> runs/p0/zr_pred/ranking_<split>.csv.
  --oof   out-of-fold class probabilities for their confident-learning filter, in their file names and format.
Test on smoke checkpoints:  --models-dir runs/p0/zr_models --alias c1=smoke_c2_fp16,... --limit-queries 50
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZR_EVAL = ROOT / "third_party" / "zr" / "code_evaluation" / "task1"
WORK = ROOT / "runs" / "p0" / "zr_src" / "predictor"
DATA = ROOT / "runs" / "p0" / "zr_data"
PRED = ROOT / "runs" / "p0" / "zr_pred"
MEMBERS = ["c1", "c2", "c3", "r1", "r2"]


def setup_predictor():
    """Their run_origin.py imports the competition starter kit (shared.base_predictor) and reads ./data/brand.txt."""
    (WORK / "shared").mkdir(parents=True, exist_ok=True)
    (WORK / "shared" / "__init__.py").write_text("")
    (WORK / "shared" / "base_predictor.py").write_text("class BasePredictor:\n    pass\n\n\nPathType = str\n")
    (WORK / "data").mkdir(exist_ok=True)
    for f in ["brand.txt", "color.txt"]:
        link = WORK / "data" / f
        if not link.exists():
            link.symlink_to(DATA / f)
    os.chdir(WORK)
    sys.path[:0] = [str(WORK), str(ZR_EVAL)]
    import run_origin  # noqa: E402  (their module; main() is guarded)
    return run_origin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--oof", action="store_true")
    ap.add_argument("--models-dir", default=str(ROOT / "runs" / "p0" / "zr_models"))
    ap.add_argument("--alias", default="", help="test only: c1=smoke_c2_fp16,r1=smoke_r1_fp16,...")
    ap.add_argument("--limit-queries", type=int, default=0, help="test only")
    ap.add_argument("--out", default=str(PRED))
    ap.add_argument("--members", default=",".join(MEMBERS), help="subset, e.g. r1 (Phase 4 parity)")
    ap.add_argument("--splits", default="dev,final_test")
    a = ap.parse_args()
    members = a.members.split(",")
    alias = dict(kv.split("=") for kv in a.alias.split(",") if kv)
    out = Path(a.out).resolve()                     # setup_predictor() chdirs into their working dir
    out.mkdir(parents=True, exist_ok=True)
    models_dir = Path(a.models_dir).resolve()

    def path(m):
        return models_dir / alias.get(m, m)

    def ready(m):
        p = path(m)
        return (p / "pytorch_model.bin").exists() and ((p / "train_time.json").exists() or m in alias)

    ro = setup_predictor()
    import numpy as np
    import pandas as pd
    from datasets import Dataset

    pred = ro.Task1Predictor()
    for m in MEMBERS:
        setattr(pred, f"model_name_or_path_{m}", str(path(m)))
    pred.model_name_or_path_c1 = str(path("c1") if ready("c1") else ROOT / "runs/p0/hf/xlm-roberta-large")  # tokenizer
    pred.prediction_setup()
    their_cls, their_reg = pred.predict_classification, pred.predict_regression

    if a.oof:
        oof = {"01": "c1", "23": "f23", "45": "f45", "67": "f67", "89": "f89"}
        for fold, m in oof.items():
            dst = DATA / "extra_confident_learnging" / f"confident_learning_task{fold}.valid_text.for_task2"
            if dst.exists():
                continue
            df = pd.read_csv(DATA / "extra_confident_learnging" / f"flod_{fold}_valid.csv")
            dl = pred.accelerator.prepare(pred.get_eval_dataloader(Dataset.from_pandas(df)))
            probs = pred.predict_classification(str(path(m)), dl)
            assert len(probs) == len(df)
            with open(dst, "w") as w:                      # the format their filter script reads
                w.write("index\tprediction\tprobs\n")
                for i, p in enumerate(probs):
                    w.write(f"{i}\t{int(np.argmax(p))}\t{','.join(str(x) for x in p.tolist())}\n")
            print(f"oof {fold} <- {m}: {len(df):,} rows", flush=True)

    if a.eval:
        timing = {}
        for split in a.splits.split(","):
            test_query_df = pd.read_csv(DATA / f"{split}.csv")
            if a.limit_queries:
                keep = test_query_df["query_id"].drop_duplicates().head(a.limit_queries)
                test_query_df = test_query_df[test_query_df["query_id"].isin(keep)]
            csv = out / f"_{split}_input.csv"
            test_query_df.to_csv(csv, index=False)
            # the same join and dataloaders as their predict()
            product_df = pd.read_csv(DATA / "catalogue_test.csv")
            test_df = pd.merge(test_query_df, product_df, how="left", left_on=["query_locale", "product_id"],
                               right_on=["product_locale", "product_id"])
            raw = Dataset.from_pandas(test_df)
            loaders = {"default": pred.get_eval_dataloader(raw), "ptbdbc": pred.get_eval_dataloader_ptbdbc(raw),
                       "ptdbbc": pred.get_eval_dataloader_ptdbbc(raw)}
            loaders = dict(zip(loaders, pred.accelerator.prepare(*loaders.values())))
            order = {"c1": "default", "c2": "ptbdbc", "c3": "ptdbbc", "r1": "default", "r2": "default"}
            outputs = {}
            for m in MEMBERS:
                f = out / f"member_{m}_{split}.parquet"
                if f.exists() and not a.limit_queries:
                    outputs[m] = pd.read_parquet(f).filter(like="s").to_numpy(dtype=np.float32)
                    continue
                if m not in members or not ready(m):
                    continue
                t0 = time.time()
                fn = their_cls if m[0] == "c" else their_reg
                o = np.asarray(fn(str(path(m)), loaders[order[m]]), dtype=np.float32).reshape(len(test_df), -1)
                timing[f"{m}_{split}_s"] = time.time() - t0
                outputs[m] = o
                cols = {f"s{i}": o[:, i] for i in range(o.shape[1])}
                pd.DataFrame({"query_id": test_df["query_id"], "product_id": test_df["product_id"], **cols}).to_parquet(f)
                print(f"{split} {m}: {len(o):,} rows in {timing[f'{m}_{split}_s']:.0f} s", flush=True)

            if all(m in outputs for m in MEMBERS):
                # their predict() end to end; its two methods hand back the member outputs computed above
                by_path = {str(path(m)): outputs[m] for m in MEMBERS}
                pred.predict_classification = lambda p, dl: by_path[p]
                pred.predict_regression = lambda p, dl: by_path[p]
                pred.predict(test_set_path=str(csv), product_catalogue_path=str(DATA / "catalogue_test.csv"),
                             predictions_output_path=str(out / f"ranking_{split}.csv"))
                pred.predict_classification, pred.predict_regression = their_cls, their_reg
                print(f"{split}: their ranking -> {out / f'ranking_{split}.csv'}", flush=True)
        prev = json.loads((out / "timing.json").read_text()) if (out / "timing.json").exists() else {}
        (out / "timing.json").write_text(json.dumps({**prev, **timing}, indent=1))


if __name__ == "__main__":
    main()
