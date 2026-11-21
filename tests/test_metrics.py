import math

import numpy as np
import pandas as pd
import pytest

from mmsearch.metrics import (by_slice, micro_f1, mrr, ndcg, ndcg_per_query, paired_bootstrap,
                              recall_at_k)


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def test_all_exact_is_perfect():
    assert ndcg(["E", "E", "E"]) == pytest.approx(1.0)


def test_substitute_and_complement_gains_are_not_swapped():
    # S ranked first: DCG = 0.1 + 0.01/log2(3); ideal is the same list.
    assert ndcg(["S", "C"]) == pytest.approx(1.0)
    # C ranked first: (0.01 + 0.1/log2 3) / (0.1 + 0.01/log2 3)
    expected = dcg([0.01, 0.1]) / dcg([0.1, 0.01])
    assert ndcg(["C", "S"]) == pytest.approx(expected)
    assert expected < 0.75  # would be ~1 if S and C had the same gain


def test_hand_computed_mixed_list():
    labels = ["I", "E", "S", "C"]
    expected = dcg([0, 1, 0.1, 0.01]) / dcg([1, 0.1, 0.01, 0])
    assert ndcg(labels) == pytest.approx(expected)
    assert ndcg(labels, k=2) == pytest.approx(dcg([0, 1]) / dcg([1, 0.1]))


def test_all_irrelevant_is_zero():
    assert ndcg(["I", "I"]) == 0.0


def frame(rows):
    return pd.DataFrame(rows, columns=["query_id", "product_id", "label", "score"])


def test_per_query_matches_single_list_and_breaks_ties_by_product_id():
    df = frame([
        (1, "b", "E", 0.5), (1, "a", "I", 0.5),   # tie -> "a" (I) first
        (1, "c", "S", 0.1),
        (2, "x", "C", 0.9), (2, "y", "E", 0.2),
        (3, "z", "I", 0.3),                        # all-I query counts as 0
    ])
    s = ndcg_per_query(df)
    assert s.loc[1] == pytest.approx(ndcg(["I", "E", "S"]))
    assert s.loc[2] == pytest.approx(ndcg(["C", "E"]))
    assert s.loc[3] == 0.0
    assert len(s) == 3


def test_per_query_rejects_nan_scores():
    with pytest.raises(ValueError):
        ndcg_per_query(frame([(1, "a", "E", float("nan"))]))


def test_paired_bootstrap():
    a = pd.Series(np.zeros(200), index=range(200))
    b = pd.Series(np.full(200, 0.01), index=range(200))[::-1]   # same queries, other order
    r = paired_bootstrap(a, b, n=1000)
    assert r["gain"] == pytest.approx(0.01)
    assert r["ci95"][0] == pytest.approx(0.01) and r["p_le_0"] == 0.0
    noise = paired_bootstrap(np.zeros(500), np.random.default_rng(1).normal(0, 1, 500), n=1000)
    assert noise["ci95"][0] < 0 < noise["ci95"][1]


def test_small_metrics():
    assert recall_at_k(["a", "b", "c"], ["c", "d"], 2) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c", "d"], 3) == 0.5
    assert mrr(["a", "b", "c"], ["b"]) == 0.5 and mrr(["a"], ["z"]) == 0.0
    assert micro_f1(["E", "S", "I"], ["E", "C", "I"]) == pytest.approx(2 / 3)
    assert by_slice([1.0, 0.0, 0.5], ["x", "x", "y"]) == {"x": {"mean": 0.5, "n": 2}, "y": {"mean": 0.5, "n": 1}}
