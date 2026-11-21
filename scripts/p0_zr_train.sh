#!/usr/bin/env bash
# Phase 0 §0.4: train one ZhichunRoad member with THEIR trainer, from a per-member working copy.
#   scripts/p0_zr_train.sh <c1|c2|c3|r1|r2|f23|f45|f67|f89> [smoke]
# Settings are those of their run_train.sh. The working copy only gets: our vocab path, and for c2/c3 the
# field-order line they left commented in code_task2/run_glue_no_trainer.py:304-305. third_party/zr is never edited.
# Env knobs: MODEL, TRAIN, VALID, NAME (override; default = Phase 0 run), EPOCHS (default 1), MIXED_PRECISION (accelerate's own switch; unset = fp32 as they ran it),
#            BS / ACCUM to shrink the per-step batch while keeping their effective batch of 256.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ZR=$ROOT/third_party/zr
DATA=$ROOT/runs/p0/zr_data
PY=$HOME/miniconda3/envs/zr/bin/python
member=$1
mode=${2:-full}

case $member in
  c1|c2|c3|f23|f45|f67|f89) task=code_task2; bs=${BS:-8};  accum=${ACCUM:-32}; seed=12345 ;;     # 4-class
  r1|r2)                    task=code_task1; bs=${BS:-16}; accum=${ACCUM:-16}; seed=12345789 ;;  # regression
  *) echo "unknown member $member"; exit 2 ;;
esac
fold=01; [[ $member == f* ]] && fold=${member#f}
train=$DATA/train_$fold.csv; [[ $member == r2 ]] && train=$DATA/train_01_cf.csv
valid=$DATA/valid_${fold}_2k.csv
train=${TRAIN:-$train}; valid=${VALID:-$valid}     # overrides (Phase 4 proxy parity)

name=$member; [[ $mode == smoke ]] && name=smoke_$member${TAG:+_$TAG}
name=${NAME:-$name}
src=$ROOT/runs/p0/zr_src/$name
out=$ROOT/runs/p0/zr_models/$name
log=$ROOT/runs/p0/zr_logs/$name.log
mkdir -p "$(dirname "$log")" "$(dirname "$src")" "$out"
rm -rf "$src"; cp -r "$ZR/$task" "$src"

if [[ $task == code_task2 ]]; then
  # Their code_task2 imports RobertaWithSampleWeight, but the repo has no 4-class copy of it there. Their evaluation
  # copy is that class with the training loss left commented out; switch on the line they used ('ce-poly-1' on the
  # labels). The 'mixed' half of that comment needs logits this class no longer returns, so it stays off.
  cp "$ZR/code_evaluation/task1/RobertaWithSampleWeight.py" "$src/"
  PYTHONNOUSERSITE=1 "$PY" - "$src/RobertaWithSampleWeight.py" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = ("        # if labels is not None:\n"
       "        #     origin_loss = util_loss_fct(logits, labels, sample_weights, self.num_labels, loss_name='ce-poly-1')\n")
new = ("        if labels is not None:\n"
       "            loss = util_loss_fct(logits, labels, sample_weights, self.num_labels, loss_name='ce-poly-1')\n")
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
EOF
fi

ORDER=default; [[ $member == c2 ]] && ORDER=ptbdbc; [[ $member == c3 ]] && ORDER=ptdbbc
DATA=$DATA ORDER=$ORDER CACHE=$ROOT/runs/p0/hf_cache PYTHONNOUSERSITE=1 "$PY" - "$src/run_glue_no_trainer.py" <<'EOF'
import os, re, sys
p = sys.argv[1]; s = open(p).read()
s = s.replace("/home/kddcup_2022/data_process/extra_vocab/", os.environ["DATA"] + "/")
# their local copies of the `datasets` metric scripts -> the same metrics by name
s = re.sub(r'load_metric\("/home/kddcup_2022/[^"]*/datasets_metric/(\w+)/\w+\.py"\)', r'load_metric("\1")', s)
# cache dirs they set in os.environ (after `import datasets`, so inert) -> ours, so nothing points at their machine
s = re.sub(r"'/home/kddcup_2022/[^']*?/\.cache/huggingface/", "'" + os.environ["CACHE"] + "/", s)
default = "item = title + ' </s></s> ' + brand + ' </s></s> ' + color + ' </s></s> ' + bullet_point + ' </s></s> ' + desc"
variant = {"ptbdbc": "item = title + ' </s></s> ' + bullet_point + ' </s></s> ' + desc + ' </s></s> ' + brand + ' </s></s> ' + color",
           "ptdbbc": "item = title + ' </s></s> ' + desc + ' </s></s> ' + bullet_point + ' </s></s> ' + brand + ' </s></s> ' + color"}
order = os.environ["ORDER"]
if order != "default":
    lines = s.split("\n")
    hits = [i for i, l in enumerate(lines) if l.strip() == default]
    assert len(hits) == 2, hits            # train preprocess + eval preprocess
    for i in hits:
        lines[i] = lines[i].replace(default, variant[order])
    s = "\n".join(lines)
left = [l for l in s.split("\n") if "/home/kddcup_2022" in l and not l.strip().startswith("#")]
assert not left, left
open(p, "w").write(s)
EOF

if [[ $mode == smoke ]]; then
  # pandas, not `head`: descriptions hold newlines inside quoted CSV fields
  PYTHONNOUSERSITE=1 "$PY" -c "import pandas as pd, sys; pd.read_csv(sys.argv[1], nrows=int(sys.argv[3])).to_csv(sys.argv[2], index=False)" \
    "$train" "$src/smoke_train.csv" "${SMOKE_ROWS:-2000}"; train=$src/smoke_train.csv
  PYTHONNOUSERSITE=1 "$PY" -c "import pandas as pd, sys; pd.read_csv(sys.argv[1], nrows=500).to_csv(sys.argv[2], index=False)" \
    "$valid" "$src/smoke_valid.csv"; valid=$src/smoke_valid.csv
fi

export PYTHONHASHSEED=12345 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=$ROOT/runs/p0/hf_datasets_cache
start=$(date +%s)
echo "[$(date '+%F %T')] $member mode=$mode train=$train valid=$valid bs=$bs accum=$accum precision=${MIXED_PRECISION:-no}" | tee "$log"
cd "$src"
"$PY" "$ROOT/scripts/gpu_cap_run.py" run_glue_no_trainer.py \
  --model_name_or_path "${MODEL:-$ROOT/runs/p0/hf/xlm-roberta-large}" \
  --max_length 256 --pad_to_max_length \
  --per_device_train_batch_size "$bs" --per_device_eval_batch_size "$bs" \
  --gradient_accumulation_steps "$accum" --learning_rate 1e-5 \
  --num_train_epochs "${EPOCHS:-1}" --output_dir "$out" --seed "$seed" \
  --train_file "$train" --validation_file "$valid" >> "$log" 2>&1
end=$(date +%s)
rows=$(PYTHONNOUSERSITE=1 "$PY" -c "import pandas as pd, sys; print(len(pd.read_csv(sys.argv[1], usecols=['query_id'])))" "$train")
echo "{\"member\": \"$member\", \"mode\": \"$mode\", \"rows\": $rows, \"seconds\": $((end - start)), \"epochs\": ${EPOCHS:-1}, \"precision\": \"${MIXED_PRECISION:-no}\", \"bs\": $bs, \"accum\": $accum}" > "$out/train_time.json"
echo "[$(date '+%F %T')] done in $((end - start)) s" | tee -a "$log"
