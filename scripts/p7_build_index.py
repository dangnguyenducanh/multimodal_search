"""Phase 7 (docs/7.md): build the first-stage indexes over the 1.2M US products and evaluate them on dev queries.
  python scripts/p7_build_index.py --build       BM25 (FTS5) + off-the-shelf bi-encoder vectors
  python scripts/p7_build_index.py --finetune    bi-encoder fine-tuned on ESCI train (Exact +, Irrelevant hard -)
  python scripts/p7_build_index.py --distill R   ... then distilled from reranker run p4/R (MarginMSE)
  python scripts/p7_build_index.py --eval        Recall@100/1000, RRF combos, latency, HNSW
  python scripts/p7_build_index.py --e2e R       retrieve -> rerank with p4/R -> condensed nDCG@10 vs judged list
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from mmsearch.esci import RUNS, SQID, load_pairs  # noqa: E402
from mmsearch.metrics import ndcg_per_query, paired_bootstrap  # noqa: E402
from mmsearch.retrieval import (BGE_PREFIX, BM25, P7, build_bm25, build_hnsw, corpus, doc_text, encode,  # noqa: E402
                                recall, rrf, topk_ip)

BASE = "BAAI/bge-small-en-v1.5"
MODELS = {"bge-small": BASE, "bge-small-ft": str(P7 / "bge-small-ft"), "bge-small-distill": str(P7 / "bge-small-distill")}


def vectors(name, p):
    f = P7 / f"vec_{name}.npy"
    if not f.exists():
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(MODELS[name], device="cuda")
        m.max_seq_length = 128
        t0 = time.time()
        np.save(f, encode(m, doc_text(p)))
        print(f"{name}: encoded {len(p):,} products in {time.time() - t0:.0f}s", flush=True)
    return np.load(f, mmap_mode="r")


def train_pairs(n_max=400_000, seed=0):
    p = load_pairs(splits_=("train",))
    from mmsearch.esci import load_p1_products
    prod = load_p1_products().set_index("product_id")
    pos = p[p["label"] == "E"].sample(frac=1, random_state=seed).head(n_max)
    neg = p[p["label"] == "I"].groupby("query_id")["product_id"].first()
    rnd = np.random.default_rng(seed).choice(prod.index.to_numpy(), len(pos))
    negs = pos["query_id"].map(neg).fillna(pd.Series(rnd, index=pos.index))
    txt = pd.Series(doc_text(prod.reset_index()), index=prod.index)
    return pd.DataFrame({"anchor": pos["query"].to_numpy(), "positive": txt.loc[pos["product_id"]].to_numpy(),
                         "negative": txt.loc[negs].to_numpy()})


def finetune(out, data, loss_name="mnrl", base=BASE):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    m = SentenceTransformer(base, device="cuda")
    m.max_seq_length = 128
    loss = losses.MultipleNegativesRankingLoss(m) if loss_name == "mnrl" else losses.MarginMSELoss(m)
    args = SentenceTransformerTrainingArguments(output_dir=str(out) + "_ckpt", num_train_epochs=1,
                                                per_device_train_batch_size=128, learning_rate=2e-5, warmup_ratio=0.05,
                                                bf16=True, logging_steps=200, save_strategy="no", report_to=[],
                                                dataloader_drop_last=True)
    qcol = "anchor" if loss_name == "mnrl" else "query"
    data = data.assign(**{qcol: [BGE_PREFIX + a for a in data[qcol]]})          # bge's query instruction
    SentenceTransformerTrainer(model=m, args=args, train_dataset=Dataset.from_pandas(data, preserve_index=False),
                               loss=loss).train()
    m.save(str(out))


def distill_data(rerank_run, n_queries=8000, k=50, seed=0):
    """Reranker scores on the fine-tuned bi-encoder's top-k for train queries -> MarginMSE triplets. The same
    unjudged scored pairs are the pseudo-label pool for Phase 4 step 7 (runs/p7/pseudo_labels.parquet)."""
    from sentence_transformers import SentenceTransformer
    from mmsearch.train import predict_run
    p = corpus()
    dv = vectors("bge-small-ft", p)
    tr = load_pairs(splits_=("train",))
    tr = tr[tr["small_version"] == 1]
    qs = tr.drop_duplicates("query_id").sample(n_queries, random_state=seed)[["query_id", "query"]]
    m = SentenceTransformer(MODELS["bge-small-ft"], device="cuda")
    ids, _ = topk_ip(encode(m, qs["query"].tolist(), query=True), np.asarray(dv), k=k)
    cand = pd.DataFrame({"query_id": np.repeat(qs["query_id"].to_numpy(), k), "query": np.repeat(qs["query"].to_numpy(), k),
                         "product_id": p["product_id"].to_numpy()[ids.ravel()]})
    judged = set(zip(tr["query_id"], tr["product_id"]))
    cand["judged"] = [x in judged for x in zip(cand["query_id"], cand["product_id"])]
    sc = predict_run("p4", rerank_run, rows=cand)
    cand["rerank"] = sc["score"].to_numpy()
    for c in ["pE", "pS", "pC", "pI"]:
        cand[c] = sc[c].to_numpy()
    cand.to_parquet(P7 / "distill_candidates.parquet", index=False)
    un = cand[~cand["judged"]]
    if un["pE"].notna().all():                                   # pseudo-labels: confident class probs, soft targets
        conf = un[un[["pE", "pS", "pC", "pI"]].max(axis=1) > 0.7]
        conf.assign(label=conf[["pE", "pS", "pC", "pI"]].idxmax(axis=1).str[1], small_version=1, locale="us")[
            ["query_id", "query", "product_id", "label", "pE", "pS", "pC", "pI", "small_version", "locale"]].to_parquet(
            P7 / "pseudo_labels.parquet", index=False)
    txt = pd.Series(doc_text(p), index=p["product_id"])
    rows = []
    for _, g in cand.groupby("query_id"):
        g = g.sort_values("rerank", ascending=False)
        top, rest = g.iloc[0], g.iloc[1:].sample(min(4, len(g) - 1), random_state=seed)
        for _, n in rest.iterrows():
            rows.append((top["query"], txt[top["product_id"]], txt[n["product_id"]], float(top["rerank"] - n["rerank"])))
    return pd.DataFrame(rows, columns=["query", "positive", "negative", "label"])


def run_lists(name, p, qs, k=1000):
    f = P7 / f"lists_{name}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return dict(zip(d["query_id"], d["ids"])), json.loads((P7 / f"latency_{name}.json").read_text())
    lat = []
    if name == "bm25":
        bm = BM25()
        out = {}
        for qid, q in zip(qs["query_id"], qs["query"]):
            t0 = time.perf_counter()
            out[qid] = bm.search(q, k)[0]
            lat.append(1000 * (time.perf_counter() - t0))
    else:
        if name == "clip-image":
            from mmsearch.images import image_table
            it = image_table()
            it = it[it.index.isin(set(p["product_id"]))]
            dv = np.stack(it["emb"].to_numpy()).astype(np.float32)
            dv = (dv / np.linalg.norm(dv, axis=1, keepdims=True)).astype(np.float16)
            ids_map = it.index.to_numpy()
            qf = pd.read_parquet(SQID / "query_features.parquet").set_index("query_id")["clip_text_features"]
            qv = np.stack(qf.loc[qs["query_id"]].to_numpy()).astype(np.float32)
            qv = (qv / np.linalg.norm(qv, axis=1, keepdims=True)).astype(np.float16)
        else:
            from sentence_transformers import SentenceTransformer
            dv = np.asarray(vectors(name, p))
            ids_map = p["product_id"].to_numpy()
            m = SentenceTransformer(MODELS[name], device="cuda")
            t0 = time.perf_counter()
            qv = encode(m, qs["query"].tolist(), query=True)
            lat.append(1000 * (time.perf_counter() - t0) / len(qs))
        t0 = time.perf_counter()
        ids, _ = topk_ip(qv, dv, k=k)
        lat.append(1000 * (time.perf_counter() - t0) / len(qs))
        out = {q: ids_map[i].tolist() for q, i in zip(qs["query_id"], ids)}
    pd.DataFrame({"query_id": list(out), "ids": list(out.values())}).to_parquet(f, index=False)
    latency = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95))} if name == "bm25" \
        else {"ms_per_query_batched": float(sum(lat))}
    (P7 / f"latency_{name}.json").write_text(json.dumps(latency))
    return out, latency


def evaluate():
    p = corpus()
    dev = load_pairs(splits_=("dev",))
    qs = dev.drop_duplicates("query_id")[["query_id", "query"]]
    rel_e = dev[dev["label"] == "E"].groupby("query_id")["product_id"].apply(list)
    rel_es = dev[dev["label"].isin(["E", "S"])].groupby("query_id")["product_id"].apply(list)
    lists, lat = {}, {}
    names = ["bm25", "bge-small"] + [n for n in ["bge-small-ft", "bge-small-distill"] if Path(MODELS[n]).exists()] + ["clip-image"]
    for n in names:
        lists[n], lat[n] = run_lists(n, p, qs)
    dense = [n for n in names if n.startswith("bge")]
    rel_e_map = rel_e.to_dict()
    best_dense = max(dense, key=lambda n: np.nanmean([recall(lists[n][q], rel_e_map.get(q, []), 100) for q in qs["query_id"]]))
    combos = {"bm25+dense": ["bm25", best_dense], "bm25+dense+image": ["bm25", best_dense, "clip-image"]}
    for c, parts in combos.items():
        lists[c] = {q: rrf([lists[x][q] for x in parts], top=1000) for q in qs["query_id"]}
    rep = {"queries": len(qs), "corpus": len(p), "latency": lat, "runs": {}}
    per = {}
    for n, L in lists.items():
        r100 = pd.Series({q: recall(L[q], rel_e.get(q, []), 100) for q in qs["query_id"]}).dropna()
        per[n] = r100
        rep["runs"][n] = {"recall@100_E": float(r100.mean()),
                          "recall@1000_E": float(np.nanmean([recall(L[q], rel_e.get(q, []), 1000) for q in qs["query_id"]])),
                          "recall@100_ES": float(np.nanmean([recall(L[q], rel_es.get(q, []), 100) for q in qs["query_id"]]))}
    for n in per:
        if n != "bm25":
            rep["runs"][n]["vs_bm25_recall@100"] = paired_bootstrap(per["bm25"], per[n].reindex(per["bm25"].index).fillna(0))
    # The image index can only hold products with a photo URL: SQID's are the judged TEST products (+ some train
    # products), so image retrieval searches a pool that already contains the answers. Its recall is optimistic;
    # it is reported but cannot win, and the end-to-end run uses bm25+dense.
    for n in rep["runs"]:
        rep["runs"][n]["optimistic_image_pool"] = "image" in n
    rep["winner"] = max((n for n in rep["runs"] if "image" not in n), key=lambda n: rep["runs"][n]["recall@100_E"])
    rep["best_dense"] = best_dense
    # HNSW for serving: recall vs exact search at efSearch values, on the winning dense model
    if best_dense in rep["runs"]:
        import faiss
        dv = np.asarray(vectors(best_dense, p)).astype(np.float32)
        fi = P7 / f"hnsw_{best_dense}.faiss"
        if not fi.exists():
            t0 = time.time()
            idx = build_hnsw(dv)
            faiss.write_index(idx, str(fi))
            rep["hnsw_build_s"] = round(time.time() - t0)
        idx = faiss.read_index(str(fi))
        from sentence_transformers import SentenceTransformer
        qv = encode(SentenceTransformer(MODELS[best_dense], device="cuda"), qs["query"].tolist()[:500], query=True).astype(np.float32)
        exact = [set(x[:100]) for x in (lists[best_dense][q] for q in qs["query_id"].tolist()[:500])]
        p_ids = p["product_id"].to_numpy()
        rep["hnsw"] = {}
        for ef in [64, 128, 256]:
            idx.hnsw.efSearch = ef
            t0 = time.perf_counter()
            _, I = idx.search(qv, 100)
            ms = 1000 * (time.perf_counter() - t0) / len(qv)
            ov = np.mean([len(e & set(p_ids[i])) / 100 for e, i in zip(exact, I)])
            rep["hnsw"][f"ef{ef}"] = {"overlap_with_exact@100": round(float(ov), 4), "ms_per_query": round(ms, 2)}
        rep["index_mb"] = round(fi.stat().st_size / 1e6)
    (P7 / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


def end_to_end(rerank_run, combo="bm25+dense"):
    """Retrieve top 100 -> rerank -> drop unjudged -> nDCG@10 ('condensed list'), vs the same reranker on the
    judged list. The gap is what retrieval loses."""
    from mmsearch.train import predict_run
    rep = json.loads((P7 / "report.json").read_text())
    p = corpus()
    dev = load_pairs(splits_=("dev",))
    qs = dev.drop_duplicates("query_id")[["query_id", "query"]]
    lists = {}
    names = combo.split("+")
    best_dense = rep.get("best_dense") or [n for n in ["bge-small-distill", "bge-small-ft", "bge-small"]
                                           if (P7 / f"lists_{n}.parquet").exists()][0]
    for n in names:
        n = best_dense if n == "dense" else ("clip-image" if n == "image" else n)
        lists[n] = run_lists(n, p, qs)[0]
    top = {q: rrf([lists[n][q] for n in lists], top=100) for q in qs["query_id"]}
    cand = pd.DataFrame([(q, qq, pid) for q, qq in zip(qs["query_id"], qs["query"]) for pid in top[q]],
                        columns=["query_id", "query", "product_id"])
    sc = predict_run("p4", rerank_run, rows=cand)
    cand["score"] = sc["score"].to_numpy()
    lab = dev[["query_id", "product_id", "label"]]
    judged = cand.merge(lab, on=["query_id", "product_id"], how="inner")         # condensed list
    full = predict_run("p4", rerank_run, "dev")
    # nDCG@10 of the retrieved+reranked list, with the IDEAL taken from the query's FULL judged list, so judged
    # products that retrieval missed count against it (a condensed list alone would hide them).
    from mmsearch.metrics import GAIN
    ideal = lab.assign(g=lab["label"].map(GAIN)).sort_values(["query_id", "g"], ascending=[True, False])
    ideal["r"] = ideal.groupby("query_id").cumcount()
    idcg = ideal[ideal["r"] < 10].assign(d=lambda d: d["g"] / np.log2(d["r"] + 2)).groupby("query_id")["d"].sum()
    got = judged.sort_values(["query_id", "score", "product_id"], ascending=[True, False, True]).copy()
    got["r"] = got.groupby("query_id").cumcount()
    dcg = got[got["r"] < 10].assign(d=lambda d: d["label"].map(GAIN) / np.log2(d["r"] + 2)).groupby("query_id")["d"].sum()
    e2e = (dcg.reindex(qs["query_id"]).fillna(0) / idcg.reindex(qs["query_id"])).fillna(0)
    ref = ndcg_per_query(full, k=10).reindex(qs["query_id"])
    rep["e2e"] = {"rerank_run": rerank_run, "retriever": combo, "ndcg@10_condensed": float(e2e.mean()),
                  "ndcg@10_judged_list": float(ref.mean()), "retrieval_loss": paired_bootstrap(ref, e2e),
                  "judged_found_in_top100_pct": round(100 * len(judged) / len(dev), 1)}
    (P7 / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["e2e"], indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--finetune", action="store_true")
    ap.add_argument("--distill", default="")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--e2e", default="")
    a = ap.parse_args()
    P7.mkdir(parents=True, exist_ok=True)
    from mmsearch import cap_gpu
    cap_gpu()
    if a.build:
        p = corpus()
        t0 = time.time()
        build_bm25(p)
        print(f"bm25 built in {time.time() - t0:.0f}s", flush=True)
        vectors("bge-small", p)
    if a.finetune and not Path(MODELS["bge-small-ft"]).exists():
        finetune(P7 / "bge-small-ft", train_pairs())
        vectors("bge-small-ft", corpus())
    if a.distill and not Path(MODELS["bge-small-distill"]).exists():
        finetune(P7 / "bge-small-distill", distill_data(a.distill), loss_name="marginmse", base=MODELS["bge-small-ft"])
        vectors("bge-small-distill", corpus())
    if a.eval:
        evaluate()
    if a.e2e:
        end_to_end(a.e2e)


if __name__ == "__main__":
    main()
