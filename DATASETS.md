# Datasets for multimodal search + recommendation

> **Decision (2026-11-21): the project uses ESCI only**, with SQID for product images. See [PLAN.md](PLAN.md).
> Everything below records the research behind that choice. LookBench, H&M, Polyvore and KAGL are **not used**.

Checked 2026-11-20. Domain is **fashion first**: both reference repos are fashion-heavy
(bhAIya runs on the Kaggle Fashion Product Images catalog, ProductLens on LookBench), and fashion
is where images matter most for search and recommendations.

## Conclusion: top 5 datasets, how to evaluate, and whether ground truth exists

All five have ground truth. Each was checked on 2026-11-21 by opening the actual label files, except H&M
(the local Kaggle token returns 401, so it was checked from Kaggle's data docs only).

| # | Dataset | What we test | Ground truth (where it comes from) | Test size (verified) | Metrics (headline first) |
|---|---|---|---|---|---|
| 1 | **LookBench** v20251201 | Query image → product (single item + whole outfit) | **Curated.** Each query has an `item_ID`; gallery images with the same `item_ID` are the matches. `category` / `main_attribute` / `other_attributes` annotations give fine, coarse and graded relevance | 2,345 queries; corpus = subset gallery (571–3,951) **+ 58,275 noise** ≈ 60k per task | **Fine Recall@1**, Coarse Recall@1, Recall@10, nDCG@10, MRR |
| 2 | **ESCI + SQID** | Text query → product (title + image) | **Human-judged.** Every query–product pair is labelled Exact / Substitute / Complement / Irrelevant | US test (small version): 8,956 queries, 181,701 judged pairs (median 16 per query) | **nDCG@10** with gains E=1, S=0.1, C=0.01, I=0; Recall@100 of Exact items for first-stage retrieval |
| 3 | **H&M** | Customer history → next-week purchases | **Behavioral.** The articles each customer actually bought in a held-out week (`t_dat, customer_id, article_id`) | Offline: last week 2020-09-16→22 (val = week before). Official: hidden week, scored via Kaggle late submission | **MAP@12** (Kaggle formula), Recall@12; reported for all / cold-start items / new customers |
| 4 | **Polyvore Outfits (disjoint)** | Bundle / "complete the look" | **Human-made outfits.** The real outfit's item is the answer; wrong candidates are random items | 15,145 FITB questions (4 candidates each); 30,290 compatibility outfits labelled 1/0 | **FITB accuracy** (random guess = 25%), **compatibility AUC** (random guess = 0.5) |
| 5 | **KAGL** (bhAIya's catalog) | Harness check: product name → image; category / color / … → products | **Derived from catalog fields.** Marqo's `gt_query_doc/*.json` (in the `marqo-FashionCLIP` repo) | Text→image: 2,000 queries (median 1 positive); fine-category: 142 queries; category: 7; sub-category: 45; color: 46; season: 4; usage: 8 | Text→image **R@1, R@10, MRR**; label tasks **P@1, P@10, MRR** |

### How each one is evaluated (and what was checked)

1. **LookBench** (image search, the headline number)
   - Protocol: embed gallery + all 9 `noise` shards together, then use pure image queries (the `image` crop; `bbox`
     marks the item in street looks). Test-only, no tuning on its queries. Score with the official
     `look-bench` evaluator (`pip install look-bench`).
   - Grading per README: *Fine* = exact category and all attributes match; *Coarse* = category matches;
     *nDCG* = graded by attribute overlap.
   - ✅ Checked: downloaded `aigen_streetlook`. 160/160 queries have ≥1 matching gallery image (1–11, median 3), and
     98% of same-item pairs share a category. Noise shards carry the same annotation columns. Note the website
     says 2,363 queries but the v20251201 data has 2,345 (`aigen_studio` 193, `real_streetlook` 981).
   - Reference: leaderboard fine R@1 overall: GR-Pro 67.38, GR-Lite 65.71, Marqo-FashionCLIP 63.24.

2. **ESCI + SQID** (text search with images)
   - Protocol, main: the official Task 1. For each test query, rerank its judged products with
     text-only, then text+image scoring, and report nDCG@10. This uses complete labels, so the numbers are fair.
   - Protocol, secondary: retrieve from all 164,900 SQID products and report Recall@100 of the Exact items.
     Unjudged products count as non-relevant, which is a lower bound, so label it that way.
   - ✅ Checked: label split E 43.9% / S 35.0% / I 16.7% / C 4.5%. 8,955 of 8,956 queries have an Exact item.
     95.2% of judged pairs have a SQID image URL, and 39/39 sampled URLs still return HTTP 200.
   - Train on the ESCI `train` split (20,888 US queries), never on `test`.
   - ⚠ Don't use the official `ranking/prepare_trec_eval_files.py`: it maps S→2 and C→3 and then scores
     `2=0.01, 3=0.1`, so Substitute and Complement gains are swapped (flagged in the SQID paper, still on `main`).
     Compute nDCG with E=1, S=0.1, C=0.01, I=0 directly.
   - Published reference numbers, same US Task-1 test set, pretrained models, no fine-tuning (SQID paper, Table 2):
     Random 0.7483 · ESCI baseline 0.8562 · SBERT text 0.8292 · CLIP-L/14 text 0.8107 · **CLIP-L/14 image 0.8225** ·
     CLIP text+image score fusion ≈ +2.4% over CLIP text. Fine-tuned text-only models reach ~0.90
     (KDD Cup 2022 winner 0.9043 private LB, but that's all 3 locales, so not directly comparable).
   - Local copy: `data/esci/` (ESCI parquet files) and `data/sqid/` (image URLs + CLIP-L/14 embeddings).
   - **KDD Cup 2022 winners** (Task 1 private LB, all 3 languages): www/NetEase 0.9043 · qinpersevere 0.9036 ·
     day-day-up/Ant 0.9035 (also won Tasks 2+3) · ZhichunRoad 0.9025 · ETS-Lab 0.9014. The top 3 published papers only.
     Public code: `wufanyou/KDD-Cup-2022-Amazon` (ETS-Lab, MIT, 2nd on Tasks 2+3) and `cuixuage/KDDCup2022-ESCI` (ZhichunRoad).
     Shared recipe, **all text-only, no images**: cross-encoder (DeBERTa-v3-large / InfoXLM / XLM-R) over
     `query [SEP] title + brand + color + bullets`, trained as 4-class E/S/C/I, ranked by expected gain
     `P(E) + 0.1·P(S) + 0.01·P(C)`, plus self-distillation for noisy labels and a multi-model ensemble.
     They trained on the `train` rows of all three tasks (1.98M pairs). Checked: 0 query–product overlap with the Task 1 test set.

3. **H&M** (recommendation)
   - Protocol: train on transactions ≤ 2020-09-15 and predict 12 articles per customer who buys in
     2020-09-16→22, then score MAP@12 + Recall@12. Customers with no purchase that week aren't scored (Kaggle rule).
   - Report separately for (a) all customers, (b) purchases of articles never sold before the cutoff
     (cold-start, where image/text embeddings must do the work), and (c) customers with no history.
   - Baselines to beat: last-week popularity, then "bought together" item-to-item.
   - ⚠ Not opened locally. `kaggle` returns **401 Unauthorized** even for `competitions list`, so the token in
     `~/.kaggle/kaggle.json` is expired or invalid. Regenerate it on kaggle.com → Settings → API and
     accept the competition rules once. Schema and date range (2018-09-20→2020-09-22) are from Kaggle's data page.

4. **Polyvore Outfits, disjoint split** (bundles)
   - FITB: given the rest of an outfit, score each of the 4 candidates (e.g. mean compatibility with the outfit
     items) and pick the best → accuracy.
   - Compatibility: score each outfit → AUC against the 1/0 labels.
   - Optional retrieval version, closer to the product feature: hide one item, rank *all* test items of
     that category, and report Recall@10 of the hidden item.
   - ✅ Checked: `owj0421/polyvore-outfits` (open, IDs + labels only) has `items`, `candidates`, `label` for FITB
     and `items`, `label` for compatibility. Images: `mvasil/polyvore-outfits` (official; needs a Hugging Face
     login and accepting its terms; disjoint test 0.66 GB) or `lzcn/outfit-datasets` (open; 9 GB LMDB).

5. **KAGL** (harness check, not a headline)
   - Protocol: run Marqo's exact query sample from `marqo-FashionCLIP/data/KAGL/gt_query_doc/` with zero-shot
     Marqo-FashionSigLIP. If you land within about 1 point of their leaderboard, the eval code is right. Then score
     bhAIya-style "LLM tags + matching" on the same files as the baseline.
   - ✅ Checked: text→image queries are product names (e.g. "Lotus Herbals … Eye Contour Definer" → 1 image).
     Label tasks have many correct items per query, so use P@k, not recall.
   - ⚠ Season (4 queries) and usage (8) are too few to compare models; use text→image and fine-category (142).

Run them in the order in **Suggested order** at the end. Add the optional sets only when a feature needs them.

## What the reference repos actually do (and what to avoid)

### bhAIya (`Shrithu10/bhAIya`, 2024, 4★)
- Catalog = `database.csv`, which is the **Kaggle Fashion Product Images** (Myntra) set
  (`id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName`)
  with a **randomly generated `price`** column, plus an Amazon text-only dump.
- "Search/recommendation" = an LLM (Mistral + LLaVA-phi) tags each product with categories, then
  Word2Vec + cosine matches query categories to product categories. Bundles and the seller dashboard
  (RFM, geo, discount clustering) run on **synthetic customers**.
- **No interaction data and no evaluation**, so none of its recommendations can be measured.
  Keep "LLM-tag + match" as a *baseline* to beat; KAGL lets you score it.

### ProductLens (`NguyenVanVuongDev/ProductLens`, 2026)
- FashionCLIP (0.7) + ViT-B/32-laion (0.3) fused to 1024-d, domain centering, "Dual-EKMA" query
  transform, metadata rerank, FAISS HNSW. Datasets: **LookBench** (fashion) and **ABO** (home/shoes/phone cases).
- Reports LookBench I2I R@1 75.5%, ABO T2I R@10 100%. **These numbers can't be compared to anything** because of four
  eval-design problems, verified in their notebooks:
  1. **Distractors dropped.** The official LookBench corpus = subset gallery + the shared 58.3k `noise` set
     (e.g. RealStudioFlat 3,951 + 58,275 = 62,226). ProductLens searches only the 571–3,951 subset images
     (the notebook never loads `noise`), a gallery 15–100× smaller.
  2. **Relevance labels leak into the ranker.** LookBench defines positives by shared category + main attribute.
     ProductLens's rerank adds `w_cat` / `w_main` bonuses when a candidate's category / main attribute equals
     the *query's annotated* ones, so it reads the answer key.
  3. **Queries aren't single-modality.** "T2I" query = 0.60·text + 0.25·metadata-hash + 0.15·**query image**;
     "I2I" query also mixes in query text + metadata hash. On ABO the T2I text is built from the target's own
     title + color + category, hence 100%.
  4. **Tuned on the benchmark.** The 2,345 LookBench queries are split 60/20/20, with transforms and τ fit on the first 80%.
     Only 468 test queries are left, and the benchmark is meant to be test-only.
- For comparison, the official leaderboard (full corpus, no tuning on queries): GR-Pro 67.38% fine R@1 overall,
  GR-Lite 65.71%, Marqo-FashionCLIP 63.24%.

**Rules this project follows:** (a) a query is exactly one modality, as a user would type or upload it;
(b) never use annotations that define relevance as query-side features; (c) official corpus incl.
distractors; (d) train/tune on other data (H&M, KAGL, Marqo-GS), use benchmarks test-only;
(e) recompute a zero-shot baseline (Marqo-FashionSigLIP) in *your* harness before claiming gains.

## Dataset details

### LookBench: image search benchmark
- 4 subsets, queries / corpus: RealStudioFlat 1,011 / 62,226 (easy), AIGen-Studio 192 / 59,254,
  RealStreetLook 1,000 / 61,553 (multi-item, hard), AIGen-StreetLook 160 / 58,846. 2.21 GB.
- Street-look subsets = "photo of an outfit → find every item": the realistic snap-and-search case
  bhAIya pitches ("Snap and Learn").
- Official eval code: `pip install look-bench` (`SerendipityOneInc/look-bench`). Since 2026-09 it also has
  **ShopRank-Bench**, a text reranking track (10,511 pairs / 2,991 live queries) you can use to test your reranker.
- Refreshed periodically, with time-stamped samples. Report which snapshot you used (v2601 at time of writing).

### ESCI + SQID: text search with real queries
- ESCI (Amazon KDD Cup 2022): real queries, up to 40 judged products each, labels Exact / Substitute /
  Complement / Irrelevant, in English, Spanish and Japanese (useful if you keep bhAIya's multilingual idea).
- SQID adds images for the US Task-1 test products: 164,900 products (95% have an image URL) + 27,139
  supplementary, plus CLIP ViT-L/14 image and text embeddings. 946 MB. Images must be fetched from Amazon URLs.
- What it answers: *does adding the product image improve text search ranking over title-only BM25 / text embeddings?*
  The SQID paper shows it does, so you have a published reference point.
- Not fashion-only (general Amazon). That's fine because it tests the text-search path, not the visual one.

### H&M: recommendation (and the catalog for the demo)
- 105,542 articles (most with an image; `product_code` groups color variants), 1,371,980 customers
  (age, club status, postal code), 31,788,324 transactions from 2018-09-20 to 2020-09-22. 34.56 GB.
- Protocol: train on everything before the last week, predict 12 articles per customer for the next 7 days,
  report MAP@12 (Kaggle formula) + Recall@12. Use the week before that as validation.
- Images and descriptions let you evaluate **cold-start** items (no purchases yet, only content embeddings)
  separately from warm items. That is the case where a multimodal model should beat collaborative filtering.
- Also usable for: "similar items" (same `product_code` = positive), text→image search (LookBench's
  leaderboard reports H&M T2I R@10/MRR@10), and seller-dashboard customer clustering with **real** customers
  instead of bhAIya's synthetic ones.
- ⚠ License: non-commercial and academic only, and you may not redistribute the data. Fine for a portfolio and
  offline benchmarks. **Don't serve H&M images from a public demo.** Use ABO or Polyvore (CC BY 4.0) there.

### Polyvore Outfits: bundles / complementary items
- 68,306 outfits / 261,058 items with images, titles, fine-grained types. Use the **disjoint** split
  (35,140 outfits, no item overlap between train and test), which the authors recommend.
- Tasks: fill-in-the-blank (pick the missing item from 4) and outfit compatibility AUC.
  Reference: Vasileva et al. type-aware embeddings reach about 0.88 AUC / 57.6% FITB (nondisjoint).
- Sources: `mvasil/polyvore-outfits` (official, gated, 4.3 GB), `owj0421/polyvore-outfits` (splits only, 160 MB),
  `Marqo/polyvore` (items only).

### KAGL: bhAIya's catalog, used as the smoke test
- 44,434 products, same columns as bhAIya's `database.csv`. `Marqo/KAGL` packages it with eval tasks:
  text→image, category / sub-category / fine-category / color / season / usage → product.
- Marqo publishes zero-shot numbers for FashionCLIP 2.0, Marqo-FashionCLIP/SigLIP, SigLIP, etc. If your
  harness reproduces them within about 1 point, the eval code is correct. Easy (studio packshots, coarse labels),
  so don't report it as the headline result.

## Optional (add only when a feature needs it)

| Feature | Dataset | Notes |
|---|---|---|
| "Like this, but red / with long sleeves" (image + text edit) | **FashionIQ** | 77,684 images, ~60k human relative captions (dress / shirt / toptee). R@10, R@50. Some image links are dead (~75k recoverable). |
| Long natural-language / chat-style queries | **Amazon-C4** | 21,223 queries over a 1,058,417-item pool (Amazon Reviews 2023). Queries are ChatGPT-rewritten 5★ reviews, so synthetic. |
| Training a multimodal ranker | **Marqo-GS-10M** | ~10M query–product pairs (~100k queries, ~5M products) with rank scores. 4 splits incl. novel-query / novel-doc / zero-shot. Train here, test on ESCI and LookBench. |
| Non-fashion / multi-view / public-demo catalog | **ABO** | 147,702 listings, 398,212 images, multilingual metadata, CC BY 4.0 since 2023. Has an official multi-view retrieval (MVR) split; use that instead of a custom one. |
| Comparing against multimodal-rec papers | **Amazon Reviews 2023** (+ MMRec Baby/Sports/Clothing) | Standard splits from the McAuley lab. Item images via URL (some dead). Interactions are reviews, a weaker signal than H&M purchases. |
| Non-product content rec (video covers) | MicroLens, PixelRec | Only if the scope moves beyond shopping. |

Skipped: DeepFashion (access by request, old), Fashion200k (noisy auto-captions), Street2Shop (dead links).

## Suggested order (simplest end-to-end first)

1. **KAGL smoke test.** Zero-shot Marqo-FashionSigLIP, reproduce the published numbers. This validates the harness.
2. **LookBench, official protocol.** Full corpus incl. `noise`, pure image queries. This is the headline image-search number.
3. **H&M recommendation.** Popularity → item-to-item co-purchase → add image/text embeddings, measured with MAP@12
   and reported separately for warm vs cold-start items.
4. **ESCI/SQID text search.** BM25 → text embeddings → + image embeddings, measured with nDCG@10.
5. **Polyvore FITB.** Score the bundle feature.
