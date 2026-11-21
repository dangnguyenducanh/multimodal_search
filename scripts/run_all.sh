#!/usr/bin/env bash
# The project's single GPU queue. User rule (2026-11-21): the whole project uses at most 25 GB of the shared card.
# So: ONE GPU job at a time, each hard-capped at MMS_GPU_GB=24 (PyTorch per-process limit; ZR's trainer through
# scripts/gpu_cap_run.py). Measured peaks: ZR large member 22.7 GiB, proxy run ~10 GiB.
# Order interleaves so results arrive early. Every step is resumable: rerun the script and finished work is skipped.
# A failed step is logged and the queue moves on.   nohup setsid scripts/run_all.sh > runs/run_all.log 2>&1 &
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
export MMS_GPU_GB=24 PYTHONNOUSERSITE=1
PY=.venv/bin/python
OURS_ALLOC=expandable_segments:True          # torch 2.11 (ours); ZR's torch 2.0 gets max_split_size_mb below

step() {
  local name=$1; shift
  echo "[$(date '+%F %T')] == $name"
  if ! "$@"; then echo "[$(date '+%F %T')] !! FAILED: $name (queue continues)"; fi
}
ours() { PYTORCH_CUDA_ALLOC_CONF=$OURS_ALLOC "$PY" "$@"; }
zr() { PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 scripts/p0_zr_queue.sh "$1"; }

step "p7 bm25 + off-the-shelf vectors"    ours scripts/p7_build_index.py --build
step "p5 CLIP features (missing products)" bash -c "[[ -f runs/p5/image_features_extra.parquet ]] || PYTORCH_CUDA_ALLOC_CONF=$OURS_ALLOC $PY scripts/p5_image_features.py --clip"
step "p5 captions"                        ours scripts/p5_image_features.py --captions
step "p7 fine-tune bi-encoder"            ours scripts/p7_build_index.py --finetune
step "p7 eval"                            ours scripts/p7_build_index.py --eval
step "p0 stage 1 (c1, r1)"                zr 1
step "p4 ladder"                          ours scripts/p4_ladder.py
step "p0 stage 2 (c2, c3)"                zr 2
step "p3 pretraining"                     ours scripts/p3_pretrain.py
CLS=$("$PY" scripts/pick_run.py --cls)
step "p7 distill + pseudo-labels ($CLS)"  bash -c "[[ -n '$CLS' ]] && PYTORCH_CUDA_ALLOC_CONF=$OURS_ALLOC $PY scripts/p7_build_index.py --distill ${CLS#p4/} && PYTORCH_CUDA_ALLOC_CONF=$OURS_ALLOC $PY scripts/p7_build_index.py --eval"
step "p4 ladder (pseudo, pretrained)"     ours scripts/p4_ladder.py
step "p5 variants A, B, C"                ours scripts/p5_multimodal.py --variants A,B,C
CLS=$("$PY" scripts/pick_run.py --cls)
step "p8 recommendations ($CLS)"          bash -c "[[ -n '$CLS' ]] && PYTORCH_CUDA_ALLOC_CONF=$OURS_ALLOC $PY scripts/p8_recs.py --run $CLS"
step "p7 end-to-end"                      bash -c "B=\$($PY scripts/pick_run.py --best); PYTORCH_CUDA_ALLOC_CONF=$OURS_ALLOC $PY scripts/p7_build_index.py --e2e \${B#p4/}"
step "p0 stage 3 (folds, CF, r2)"         zr 3
step "p4.2 full scale"                    ours scripts/p4_full.py
BEST=$("$PY" scripts/pick_run.py --best); FULL=$("$PY" scripts/pick_run.py --full)
IMGS=$(ls -d runs/p5/caption-field runs/p5/image-in-head 2>/dev/null | sed 's|runs/||' | paste -sd, -)
step "p6 ensemble"                        ours scripts/p6_ensemble.py --ours "$BEST${FULL:+,$FULL}" --images "$IMGS"
step "p9 demo tables"                     ours scripts/p9_add_models.py
echo "[$(date '+%F %T')] run_all finished"
