# multimodal_search

Multimodal product search (text and image) + item-to-item recommendations, built and measured on **ESCI only**
(Amazon Shopping Queries, with SQID's product images). The baseline is the KDD Cup 2022 5th-place (ZhichunRoad) code,
run as-is in Phase 0. Every later phase improves on it, mainly with the 1st-place (www) recipe and our own image,
retrieval and recommendation work. End-to-end spec: [PLAN.md](PLAN.md). Dataset research: [DATASETS.md](DATASETS.md).

## Layout

✅ exists · each other entry is tagged with the PLAN.md phase that creates it

```
multimodal_search/
├── README.md               layout + how to run
├── PLAN.md              ✅ phases 0–10 overview: what to build, where it comes from, how it's judged
├── docs/                ✅ one spec per phase, 0.md … 10.md: their method → weaknesses → our target → our proposal
│                           (0 baseline, 1 data, 2 preprocessing, 3 pretraining, 4 reranker, 5 images, 6 ensemble,
│                            7 retrieval, 8 recommendations, 9 backend, 10 frontend)
├── third_party/zr/      ✅ ZhichunRoad's repo @ 301bb98, cloned for the Phase 0 baseline; no license → gitignored, never edited
├── DATASETS.md          ✅ dataset research; ESCI chosen
├── RESULTS.md         ✅ generated from runs/*/*/metrics.json (scripts/update_results.py): dev nDCG ± CI, gain vs ZR, GPU-h, ms/query
├── pyproject.toml       P0  dependencies; `pip install -e .`
├── .gitignore           ✅ keeps data/, runs/ and third_party/ out of git
│
├── data/                   raw downloads, read-only, never committed
│   ├── esci/            ✅ labels (2.6M judged pairs) + products (1.8M, 1.2M US)
│   ├── sqid/            ✅ image URLs + CLIP-L/14 text/image embeddings (US)
│   └── images/          P2  product photos fetched from SQID URLs
│
├── mmsearch/               the library; scripts/ and app/ import from here
│   ├── metrics.py     ✅ P0  nDCG (E=1, S=0.1, C=0.01, I=0), Recall@k, MRR, F1, paired bootstrap
│   ├── esci.py        ✅ P0  split(), query_slices() · P1 joins, query-hash folds, sample weights
│   ├── text.py          P2  cleaning, product-text formats A/B, field orders, vocabularies
│   ├── augment.py       P2  translation, C/I rebalancing, confident-learning filter
│   ├── pretrain.py      P3  masked words + Product2Query + brand/color heads
│   ├── model.py         P4  cross-encoder, 4-class + regression heads, extra-feature block (P5: image block)
│   ├── train.py         P4  training loop: self-distillation, label smoothing, AWP/FGM, R-Drop, EMA
│   ├── ensemble.py    ✅ P0  zr_score() (ZR's formula, parity-checked) · P6 expected gain, weights, stacking, pruning
│   ├── retrieval.py     P7  BM25, text vectors, CLIP image vectors, rank fusion (FAISS)
│   └── recs.py          P8  alternatives (P(S)) and goes-with (P(C))
│
├── scripts/                one entry point per phase step, each writes runs/<phase>/<run>/
│   └── ✅ p0_check_harness.py, p0_zr_data.py, p0_zr_train.sh, p0_zr_queue.sh, p0_zr_infer.py, p0_score_zr.py, update_results.py … p9_export.py
│
├── runs/                   everything generated: tables, embeddings, checkpoints, predictions, metrics.json
├── tests/
│   └── test_metrics.py ✅ P0  metrics checked against hand-computed cases
├── app/                 ✅ FastAPI backend (BM25 only until training; docs/demo-handoff.md)
└── web/                 ✅ React + Vite + TS frontend, live at https://esci-mmsearch.vercel.app
```

## Rules

1. **`data/` is raw and read-only.** Anything derived (cleaned tables, embeddings, captions, checkpoints) goes to `runs/`.
2. **One scoring path:** `mmsearch/metrics.py` (tested), never ESCI's bundled eval script (it swaps Substitute/Complement).
3. **Splits are fixed in Phase 0** (`split()`, reused by Phase 1): train (all US train rows), dev (2,000 test queries), final test (the other 6,956).
4. **Decisions on dev, reporting on final test.** "Better" = paired-bootstrap 95% CI of the gain above 0.
5. **The ZR baseline is the bar.** Every RESULTS.md row shows its gain vs the Phase 0 ZR baseline. Their code is run
   from `third_party/zr/` (or a per-run working copy) and never copied into our files: no license.
6. **Every run is traceable:** `metrics.json` records arguments, models, data version, GPU-hours and latency;
   `RESULTS.md` only copies from these files.
