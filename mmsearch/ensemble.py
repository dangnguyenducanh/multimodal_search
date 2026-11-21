"""Ensemble scoring (docs/6.md). Phase 0 needs only zr_score(); Phase 6 adds the improvements."""
import numpy as np
import pandas as pd

# ZhichunRoad's label ids come from sorting the label words: complement 0, exact 1, irrelevant 2, substitute 3.
ZR_LABEL = np.array(["C", "E", "I", "S"])
ZR_GAIN = np.array([0.01, 1.0, 0.0, 0.1])


def zr_score(class_probs=(), reg_scores=()):
    """Their run_origin.py sort_key_4, reimplemented (their repo has no license):
    p = mean of the classifiers' probabilities; score = gain(argmax p) * max p, plus the mean regression score.
    With no regressors this is their sort_key_3; with no classifiers, the mean regression score.
    Same dtypes as theirs (float32 means, float64 products), so rankings match theirs exactly."""
    s = 0.0
    if len(class_probs):
        p = np.mean(np.stack([np.asarray(c, np.float32) for c in class_probs]), axis=0)
        top = p.argmax(axis=1)
        s = ZR_GAIN[top] * p[np.arange(len(p)), top].astype(np.float64)
    if len(reg_scores):
        s = s + np.mean(np.stack([np.asarray(r, np.float32).reshape(-1) for r in reg_scores]), axis=0).astype(np.float64)
    return s


# ---- Phase 6 stages (docs/6.md) --------------------------------------------------------------------------------
GAIN_ESCI = np.array([1.0, 0.1, 0.01, 0.0])                       # our order E, S, C, I


def expected_gain(probs):
    return np.asarray(probs) @ GAIN_ESCI


def rank_norm(score, query_ids):
    """Per-query percentile rank, so members on different scales can be averaged (stage 1)."""
    return pd.Series(score).groupby(np.asarray(query_ids)).rank(pct=True).to_numpy()


def weighted(table, members, w):
    return sum(w[m] * table[m].to_numpy() for m in members)


def fit_weights(table, members, ndcg_fn, rounds=3, grid=np.linspace(0, 2, 21)):
    """[www]-style weights by coordinate ascent on nDCG (start: equal weights)."""
    w = {m: 1.0 for m in members}
    best = ndcg_fn(weighted(table, members, w))
    for _ in range(rounds):
        for m in members:
            for v in grid:
                trial = {**w, m: v}
                if sum(trial.values()) == 0:
                    continue
                nd = ndcg_fn(weighted(table, members, trial))
                if nd > best + 1e-6:
                    best, w = nd, trial
    return w, best


def greedy_prune(members, score_fn, cost, tol=0.0):
    """Stage 7: drop members one at a time while the score doesn't fall by more than tol; costliest first on ties."""
    keep = list(members)
    base = score_fn(keep)
    while len(keep) > 1:
        trials = sorted(((score_fn([m for m in keep if m != d]), -cost.get(d, 0), d) for d in keep), reverse=True)
        s, _, d = trials[0]
        if s < base - tol:
            break
        keep.remove(d)
        base = s
    return keep, base
