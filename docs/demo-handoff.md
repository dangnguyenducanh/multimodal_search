# Web demo handoff (Phases 9–10): what's live, what's blank, how to plug in trained models

**Live:** https://esci-mmsearch.vercel.app (Vercel project `esci-mmsearch`, Hobby plan, deployed by CLI, no git link).
**Built 2026-11-21** before any training. Every model slot is deliberately empty; nothing on the page claims a
model result. This doc is for the agent that finishes Phases 4–8.

## What exists

| Piece | Where | State |
|---|---|---|
| Demo DB | `scripts/p9_build_db.py` → `runs/p9/catalog.sqlite` (134 MB) | built from `data/` directly (Phase 1 outputs don't exist yet) |
| Backend | `app/` (FastAPI) | spec'd endpoints from 9.md (REST API); BM25 only |
| Frontend | `web/` (React 19 + Vite 7 + TS strict) | Home / Search / Product / 404 |
| Tests | `tests/test_api.py` | contract smoke test, `python -m pytest -q tests/test_api.py` |
| Deploy | `scripts/p9_vercel_deploy.sh` | builds web, bundles to `runs/p9/esci-mmsearch/`, `vercel deploy --prod` |

**Catalog** = the 164,900 US Task 1 **test** products (judged for the 8,956 dev + final-test queries), 156,545 with
a SQID image URL (hotlinked from Amazon's CDN). Not the full 1.2M US catalog: the full one doesn't fit a Vercel
function. Bullets are truncated to 250 characters; descriptions are dropped.

**DB tables** (9.md (Database) plus one): `products` (+ `products_fts` FTS5, weights title 10 / brand 4 / color 2 /
bullets 1), `queries` (all `split='test'`; Phase 1 decides dev vs final test), `judgments`, `examples` (20 random
queries, `reason='random'`), `models` (**empty**), `looks_like` (8 nearest products by SQID CLIP image cosine, as
`"rowid:cos,…"`; no training involved). `scripts/p9_add_models.py` fills the model tables (below).

## Deviations from 9.md / 10.md (on purpose)

| Spec | Demo today | Why |
|---|---|---|
| `/api/search` retrieve → rerank | retrieve (BM25 FTS5 top 100) → rerank only for known dev/test queries (precomputed scores) | a cross-encoder doesn't fit a Vercel function |
| `/api/search/image`, `/photo` page | endpoint returns 503 `MODEL_NOT_READY`; page dropped | needs a CLIP image encoder at request time, too big for Vercel |
| — | **new** `GET /api/products/{id}/looks-like` + "Looks similar" row | stands in for photo search; uses precomputed SQID embeddings |
| `/alternatives`, `/goes-with` | precomputed `recs` for the example queries' products (empty list for others); 503 `MODEL_NOT_READY` until the table exists | need P(S)/P(C) from the Phase 8 reranker; no model on Vercel |
| `ProductHit.probs/predicted/rank_text_only` | `null` for free-text queries (precomputed rerankers cover known queries only) | no model on Vercel |
| `SearchResponse` | adds `retriever`, `reranker` (the model's name, or null) | lets the UI say honestly what ranked the results |
| ModeToggle | disabled while `reranker` is null | nothing to toggle |
| lifespan loading, ONNX, LRU cache, request-id middleware, CORS | not built | nothing to load; Vite proxy covers dev |
| web `components/`, `pages/` folders | `src/components.tsx`, `src/pages.tsx` | small enough for one file each |
| DB shipped as file | shipped as `catalog.sqlite.gz`, unpacked to `/tmp` on cold start (`app/db.py: db_file`) | Hobby plan caps uploads at 100 MB per file |

## How to plug in trained models

`scripts/p9_add_models.py` (the last step of `scripts/run_all.sh`) writes the model tables into the demo DB
(`--db` for another path). It is partial by design: each part runs once its inputs exist, prints what it skipped,
and a rerun refreshes everything it can.

| Table | From | Used by |
|---|---|---|
| `models` (+ `ms_per_query`) | each run's `metrics.json` | `/api/models`, footer |
| `pair_scores(query_id, product_id, model, score, pE, pS, pC, pI)` | every dev + final-test pair of: ZR's 5 members (`runs/p0/zr_pred`, `zr_score`) as role `zr`; our text winner (`pick_run.py --full`, else `--best`) as `ours`; the better Phase 5 run as `multimodal` | rerankers, `/api/compare` |
| `examples` | 10 gained + 5 lost + 5 random dev queries by per-query nDCG gain of `ours` over `zr` | example chips |
| `recs` | the `pick_run.py --cls` run's P(S) / P(C) over BM25's top 50 by title, for the products in the examples' judged lists; top 10 each (GPU, `--no-recs` skips) | Alternatives / Goes with |

`pair_scores` replaces 9.md's `baseline_scores` (one row per model, so any model can be added). Three models add
~45 MB to the DB (measured: 61 → 78 MB gzipped), still under Vercel's 100 MB per-file cap.

Rerankers (`app/services/rerank.py`, installed by `app/services/pipeline.py`):
- **Precomputed (default, fits Vercel):** a known dev/test query gets its judged products ranked by the saved
  scores of role `ours` (text mode) or `multimodal` (falls back to `ours`). Free-text queries stay BM25 only
  (`reranker: null`). `rank_text_only` is filled when the two modes use different models.
- **Live (GPU host):** `MMS_RERANKER=runs/p4/<run>` (text) and optionally `MMS_MM_RERANKER=runs/p5/<run>` load the
  model once at startup and rerank BM25's top 100 for any query; torch is imported only then. GPU budget: the
  project's GPU jobs together stay under 25 GB; Live caps itself at `MMS_GPU_GB` (default 8) and the queue takes 24,
  so run it while the queue is idle. A live host is also what photo upload and the full catalog (Phase 7) need.

Then `bash scripts/p9_vercel_deploy.sh`.

## Run locally
```bash
python scripts/p9_build_db.py          # ~20 s, GPU for look-alikes
python scripts/p9_add_models.py        # model tables (skips what isn't trained yet)
cd web && npm install && npm run build && cd ..
uvicorn app.main:app --port 8000       # serves web/dist at / and the API at /api
```
Dev with hot reload: `uvicorn app.main:app --port 8000 --reload` plus `cd web && npm run dev` (port 5173, proxies `/api`).

## Known limits
- BM25 with OR-ed tokens: color/attribute words are not enforced (e.g. "red dress" returns any dress), which is
  exactly the gap the rerankers are meant to close.
- About 5% of products have no image (placeholder shown, no look-alikes).
- Cold start: about 0.5 s extra to unpack the DB.
