#!/usr/bin/env bash
# Phase 0 §0.4–0.6, the whole run order, resumable (a member is done when its train_time.json exists):
#   1. c1 + r1 -> score    2. c2, c3 -> score    3. f23..f89 -> their confident-learning filter -> r2 -> score
# fp16 through accelerate's own MIXED_PRECISION switch: 2.7x faster than their fp32 in the smoke run (deviation,
# logged in RESULTS.md). Stages so the project queue (scripts/run_all.sh) can interleave other GPU work:
#   scripts/p0_zr_queue.sh 1|2|3   (no argument = all three)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
ZPY=$HOME/miniconda3/envs/zr/bin/python
OPY=$HOME/.conda/envs/jobmatch/bin/python
M=runs/p0/zr_models
D=runs/p0/zr_data
export MIXED_PRECISION=fp16 PYTHONNOUSERSITE=1

done_() { [[ -f $M/$1/train_time.json ]]; }
train() {
  done_ "$1" && return 0
  echo "[$(date '+%F %T')] train $1"
  scripts/p0_zr_train.sh "$1"
  rm -rf runs/p0/hf_datasets_cache          # ~6-9 GB of Arrow/tokenized cache per member; disk is tight
}
score() {
  echo "[$(date '+%F %T')] infer + score"
  "$ZPY" scripts/p0_zr_infer.py --eval
  "$OPY" scripts/p0_score_zr.py
}

stage=${1:-all}
if [[ $stage == 1 || $stage == all ]]; then train c1; train r1; score; fi
if [[ $stage == 2 || $stage == all ]]; then train c2; train c3; score; fi
[[ $stage == 3 || $stage == all ]] || exit 0

for f in 23 45 67 89; do
  if ! done_ "f$f"; then
    [[ -f $D/train_$f.csv ]] || PYTHONHASHSEED=12345 "$ZPY" scripts/p0_zr_data.py --folds "$f"
    train "f$f"
  fi
  rm -f "$D/train_$f.csv"
done

if ! done_ r2; then
  "$ZPY" scripts/p0_zr_infer.py --oof          # out-of-fold probabilities, their file names
  # their filter script, unchanged except its hard-coded row total (2,069,039 in their data) -> ours
  n=$("$ZPY" -c "import pandas as pd; print(sum(len(pd.read_csv('$D/extra_confident_learnging/flod_'+f+'_valid.csv', usecols=['query_id'])) for f in ['01','23','45','67','89']))")
  mkdir -p runs/p0/zr_src/cf
  sed "s/2069039/$n/" third_party/zr/code_preprocess/7_confidient_learning_filter.py > runs/p0/zr_src/cf/filter.py
  (cd "$D" && PYTHONHASHSEED=12345 "$ZPY" "$ROOT/runs/p0/zr_src/cf/filter.py")
  PYTHONHASHSEED=12345 "$ZPY" scripts/p0_zr_data.py --cf-filter
  train r2
fi
score
echo "[$(date '+%F %T')] queue finished"
