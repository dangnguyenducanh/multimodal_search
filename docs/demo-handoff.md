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
`"rowid:cos,…"`; no training involved).

## Deviations from 9.md / 10.md (on purpose)

| Spec | Demo today | Why |
|---|---|---|
| `/api/search` retrieve → rerank | retrieve (BM25 FTS5 top 100) → **no rerank** | no trained model |
| `/api/search/image`, `/photo` page | endpoint returns 503 `MODEL_NOT_READY`; page dropped | needs a CLIP image encoder at request time, too big for Vercel |
| — | **new** `GET /api/products/{id}/looks-like` + "Looks similar" row | stands in for photo search; uses precomputed SQID embeddings |
| `/alternatives`, `/goes-with` | 503 `MODEL_NOT_READY` (page shows "Coming after training") | need P(S)/P(C) from the Phase 8 reranker |
| `ProductHit.probs/predicted/rank_text_only` | always `null` | no model |
| `SearchResponse` | adds `retriever`, `reranker` (null) | lets the UI say honestly what ranked the results |
| ModeToggle | disabled while `reranker` is null | nothing to toggle |
| lifespan loading, ONNX, LRU cache, request-id middleware, CORS | not built | nothing to load; Vite proxy covers dev |
| web `components/`, `pages/` folders | `src/components.tsx`, `src/pages.tsx` | small enough for one file each |
| DB shipped as file | shipped as `catalog.sqlite.gz`, unpacked to `/tmp` on cold start (`app/db.py: db_file`) | Hobby plan caps uploads at 100 MB per file |

## How to plug in trained models (in order)

1. **Scores and labels in the page:** after Phase 4/5/6 runs exist, insert rows into `models`
   (`name, role ('text'|'multimodal'), dev_ndcg, ci_low, ci_high, test_ndcg, built_at`) inside
   `scripts/p9_build_db.py`, read from the runs' `metrics.json` (never typed by hand, 10.md (Eval) "honesty").
   The footer shows them automatically.
2. **Examples:** replace the random pick with 10 `gained` + 5 `lost` + 5 `random` queries, ranked by the paired
   per-query nDCG gain of multimodal vs text-only on **dev** (harness `per_query.parquet`). Chips are already tagged.
3. **Reranker:** set `app/services/pipeline.py: RERANKERS["multimodal"]` and `["text"]` to callables
   `(query, hits) -> hits` that fill `score`, `probs`, `predicted`, and re-sort. For the toggle arrows, run both and
   fill `rank_text_only`. The UI enables the toggle and switches the score label as soon as `reranker` is non-null.
4. **Where the reranker runs:** a cross-encoder won't fit a Vercel Python function (250 MB unzipped; torch/ONNX
   Runtime + weights exceed it). Pick one:
   - **Precompute (fits Vercel):** score the judged lists for all 8,956 test queries offline, store
     `(query_id, product_id, score, pE, pS, pC, pI)` per model in a new table, and have the reranker callable look
     them up. Free-text queries stay BM25 only; say so in the UI.
   - **Live model:** run `app/` on a GPU host (this machine or a cloud VM) and point the frontend at it
     (Vercel rewrite of `/api/(.*)` to that host). This is also the only way to bring back photo upload.
5. **Alternatives / Goes with (Phase 8):** same choice as step 4. Precompute the top 10 by P(S) and by P(C) per
   product into tables like `looks_like`, then replace `needs_reranker` in `app/routers/products.py` with lookups.
6. **Full catalog (Phase 7):** a 1.2M-product index needs the live-model host; the Vercel build stays on the
   test-product subset.
7. Update `tests/test_api.py` (drop the `MODEL_NOT_READY` and `models == []` asserts), rebuild the DB, run
   `bash scripts/p9_vercel_deploy.sh`.

## Run locally
```bash
python scripts/p9_build_db.py          # ~20 s, GPU for look-alikes
cd web && npm install && npm run build && cd ..
uvicorn app.main:app --port 8000       # serves web/dist at / and the API at /api
```
Dev with hot reload: `uvicorn app.main:app --port 8000 --reload` plus `cd web && npm run dev` (port 5173, proxies `/api`).

## Known limits
- BM25 with OR-ed tokens: color/attribute words are not enforced (e.g. "red dress" returns any dress), which is
  exactly the gap the rerankers are meant to close.
- About 5% of products have no image (placeholder shown, no look-alikes).
- Cold start: about 0.5 s extra to unpack the DB.
