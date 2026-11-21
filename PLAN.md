# Plan: improve on a real baseline, end to end

**Dataset: ESCI only** (+ SQID image links / CLIP embeddings). Target locale: US English, because SQID images are
US-only. ES/JP rows are optional extra *training* data, kept only if they help US (Phase 1).

**The baseline is ZhichunRoad's own code** (KDD Cup 2022 Task 1, 5th place, 0.9025 private), cloned into
`third_party/zr/`, retrained and run on our splits in Phase 0, and scored with our own tested nDCG. Every later phase
states **their method → its weaknesses → our target → our proposal**, and is judged by its gain over that baseline.

**Source tags:**
- **[ZR]**: ZhichunRoad. Code github.com/cuixuage/KDDCup2022-ESCI @ `301bb98` (no license), paper amazonkddcup.github.io/papers/4952.pdf.
- **[www]**: 1st place, 0.9043 private. Paper arXiv 2208.02958, no code. The main source of improvements over ZR.
- **[ours]**: images, listwise loss, retrieval, recommendations, serving and the demo.

**One spec per phase:** [0](docs/0.md) baseline · [1](docs/1.md) data · [2](docs/2.md) preprocessing ·
[3](docs/3.md) pretraining · [4](docs/4.md) reranker · [5](docs/5.md) images · [6](docs/6.md) ensemble ·
[7](docs/7.md) retrieval · [8](docs/8.md) recommendations · [9](docs/9.md) backend · [10](docs/10.md) frontend.

## The system end to end

```
OFFLINE
 0 ZR baseline (their code) ─────────────── the bar every phase must beat
 1 data ─▶ 2 preprocessing ─▶ 3 pretraining (if it pays) ─▶ 4 reranker ─▶ 5 + images ─▶ 6 ensemble
                                                                                          │
 7 retrieval index (BM25 + text vectors + image vectors over 1.2M US products) ◀──────────┘
ONLINE
 query (text or photo) ─▶ 7 retrieve top 100 ─▶ 6 rerank ─▶ results ─▶ 8 alternatives / goes with ─▶ 9 API ─▶ 10 demo
```

## How every phase is judged

| Split | What | Size (US) | Used for |
|---|---|---|---|
| train | all `train` rows, Task 1 + Tasks 2/3 | 74,888 queries · 1,393,063 pairs | training (internal folds split by query) |
| dev | 2,000 queries drawn once from the Task 1 test set | ≈ 40k pairs | every decision: checkpoints, ladder steps, weights |
| final test | the other 6,956 Task 1 test queries | ≈ 141k pairs | reporting a phase winner (and the ZR baseline), nothing else |

Dev comes from test because only 260 train queries have images for ≥ 90% of their products (test queries: 95%).

- **Metric:** nDCG over each query's judged products, gains E=1 · S=0.1 · C=0.01 · I=0, from our own tested code
  (ESCI's eval script swaps S and C).
- **"Better"** = paired bootstrap over dev queries (10,000 resamples); the 95% CI of the gain must sit above 0.
  Otherwise it's a tie, and the simpler or cheaper option stays.
- **Proxy scale** ([4.md](docs/4.md) §4.0): changes in Phases 1–5 are tested cheaply on `xlm-roberta-base`, Task 1 US
  rows, 1 epoch, against "ZR recipe @ proxy" (our reimplementation, parity-checked against their code). Winners are
  confirmed at full scale against the Phase 0 numbers.
- **Every row in RESULTS.md:** dev nDCG ± CI, **gain vs ZR baseline**, gain vs the current best, slices (query length ·
  visual-attribute queries · has image / no image), GPU-hours to train, ms per query to serve. Final test only for
  phase winners.
- **Anchors** (SQID paper, US test): random 0.7483 · CLIP text 0.8107 · CLIP image 0.8225 · ESCI official baseline
  0.8562 · **ZR baseline: measured in Phase 0**. Competition scores (0.90+) cover 3 languages on hidden splits, so
  they are not directly comparable.

## Phases at a glance

| # | Their method [ZR] | Main weakness | Our target | Our proposal (headline) |
|---|---|---|---|---|
| 0 | released code: 3 classifiers + 2 regressors, XLM-R-large, `sort_key_4` | no checkpoints, competition-era paths/format | run it unchanged on our data; ZR baseline nDCG ± CI on dev + final test | clone @ `301bb98`, adapters only (data columns, paths, torch for sm_89); our scorer passes SQID anchors first |
| 1 | all locales, `hash(query) % 10` folds, validation from train | folds not reproducible; validation unlike test | deterministic splits; all comparisons on the same test-like dev | md5 folds; dev/final test from Task 1 test; US-only vs +ES/JP measured |
| 2 | format B, 256 padded, translation, rebalance, CF filter | truncation unmeasured; padding waste; augmentations never tested alone | ≥ same nDCG for fewer GPU-hours; each augmentation earns its place | `only_second` truncation, dynamic padding, field budgets, format A [www], each ZR step as a switch |
| 3 | multi-task pretraining (MLM + Product2Query + brand/color), test products oversampled | +0.11 F1 pts for 10 large-model epochs; transductive | keep only if it beats vanilla at a fair GPU-hour cost | their code at proxy (no test oversampling); cheap version; real queries as Product2Query |
| 4 | XLM-R cross-encoder, CE/MSE, FGM + R-Drop + EMA | multilingual backbone, pointwise loss, missing [www] gains, 2× cost tricks unmeasured | one model beats ZR r1, ideally the ZR ensemble | expected-gain score, listwise loss, trick ablation, AWP, smoothing, self-distillation, DeBERTa-v3-large |
| 5 | text only (no images in 2022); brand/color ID side features | blind to appearance | significant gain, biggest on visual queries, no loss without image | late fusion → combiners, caption field, image embedding in their side-feature slot |
| 6 | 5 members, argmax-gain × prob + raw regression sum, equal weights | discards probability mass; one model family; 5× cost | beat ZR ensemble on dev + final test; report cost | expected gain, fitted weights, family diversity, image members, stacking, pruning |
| 7 | none (ranks given candidates) | can't search | Recall@100 above BM25; small end-to-end loss | BM25 + fine-tuned/distilled bi-encoder + CLIP image, RRF |
| 8 | Task 3 substitute model, query–product only | nothing product-to-product; no complements | beat random/text/image baselines; Task 3 F1 ≥ ZR's | P(S) / P(C) from one reranker + image similarity |
| 9 | batch script, models loaded per run, FP32, 256 padding | no API; far from interactive latency | parity with offline scores; p95 within budget | FastAPI, models loaded once, ONNX FP16, pre-tokenized text, `/compare` |
| 10 | none | improvement invisible outside a table | ZR vs ours visible per query, numbers from RESULTS only | shop-style React demo with a Baseline-vs-ours page |

## Order of work
0 (scorer → ZR baseline) → 1 → 4.0 (our reimplementation, parity at proxy) → 2 (switches at proxy) → 4.1 ladder
(pseudo-labels use BM25 candidates until Phase 7 exists) → 3 (if time) → 4.2 full scale → 5 → 6 → 7 → 8 → 9 → 10.
Each phase starts from the previous phase's dev winner, and every winner is reported against the ZR baseline.
