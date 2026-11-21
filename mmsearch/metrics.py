"""The one scoring path for every phase (docs/0.md §0.1).

Gains E=1, S=0.1, C=0.01, I=0. ESCI's own prepare_trec_eval_files.py swaps S and C, so it is never used.
A query whose judged products are all Irrelevant has no ideal DCG: its nDCG is defined as 0 and it still counts
(there are none in the Task 1 US test set, checked 2026-11-21).
"""
import numpy as np
import pandas as pd

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def ndcg(labels_in_ranked_order, k=None):
    """nDCG of one ranked list of E/S/C/I labels. k=None scores the whole judged list (Task 1 setting)."""
    g = np.array([GAIN[x] for x in labels_in_ranked_order], dtype=float)
    ideal = np.sort(g)[::-1]
    if k is not None:
        g, ideal = g[:k], ideal[:k]
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    idcg = float(ideal @ disc)
    return float(g @ disc) / idcg if idcg > 0 else 0.0


def ndcg_per_query(df, score="score", k=None):
    """df: query_id, product_id, label (E/S/C/I) and a score column -> Series query_id -> nDCG.

    Ranked by score descending; ties broken by product_id ascending, so a run is deterministic.
    """
    q = df["query_id"].to_numpy()
    gain = df["label"].map(GAIN).to_numpy(dtype=float)
    pid = pd.factorize(df["product_id"], sort=True)[0]
    s = df[score].to_numpy(dtype=float)
    if np.isnan(s).any():
        raise ValueError(f"{np.isnan(s).sum()} NaN scores; give missing items an explicit score")

    def dcg(order):
        qs, gs = q[order], gain[order]
        rank = pd.Series(qs).groupby(qs).cumcount().to_numpy()
        contrib = gs / np.log2(rank + 2)
        if k is not None:
            contrib = np.where(rank < k, contrib, 0.0)
        return pd.Series(contrib).groupby(qs).sum()

    d = dcg(np.lexsort((pid, -s, q)))
    i = dcg(np.lexsort((-gain, q)))
    return (d / i.where(i > 0)).fillna(0.0).rename("ndcg").rename_axis("query_id")


def paired_bootstrap(a, b, n=10_000, seed=0):
    """Per-query scores of two runs on the same queries -> gain of b over a, 95% CI, p(gain <= 0)."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        if set(a.index) != set(b.index):
            raise ValueError("runs cover different queries")
        b = b.loc[a.index]
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), (m, len(d)))].mean(axis=1)
                            for m in [500] * (n // 500) + [n % 500] if m])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"gain": float(d.mean()), "ci95": [float(lo), float(hi)], "p_le_0": float((means <= 0).mean())}


def mean_ci(x, n=10_000, seed=0):
    """Mean of per-query scores with its bootstrap 95% CI."""
    r = paired_bootstrap(np.zeros(len(x)), np.asarray(x, dtype=float), n, seed)
    return {"mean": r["gain"], "ci95": r["ci95"]}


def recall_at_k(ranked_ids, relevant_ids, k):
    rel = set(relevant_ids)
    return len(rel & set(list(ranked_ids)[:k])) / len(rel) if rel else float("nan")


def mrr(ranked_ids, relevant_ids):
    rel = set(relevant_ids)
    return next((1.0 / (i + 1) for i, x in enumerate(ranked_ids) if x in rel), 0.0)


def micro_f1(pred, gold):
    """Micro-F1 over single-label predictions, which equals accuracy (the KDD Cup Task 2/3 metric)."""
    return float(np.mean(np.asarray(pred) == np.asarray(gold)))


def by_slice(scores, slice_labels):
    """Per-query scores + a slice label per query (same index) -> {slice: {"mean", "n"}}."""
    g = pd.Series(np.asarray(scores, dtype=float)).groupby(np.asarray(slice_labels))
    return {str(k): {"mean": float(v.mean()), "n": int(v.size)} for k, v in g}
