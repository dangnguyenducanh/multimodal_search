import numpy as np
import pandas as pd
import pytest

from mmsearch.ensemble import expected_gain, fit_weights, greedy_prune, rank_norm, weighted, zr_score


def test_zr_score_matches_their_formula_by_hand():
    # their label order: complement, exact, irrelevant, substitute
    c1 = np.array([[0.1, 0.45, 0.0, 0.45], [0.0, 0.2, 0.8, 0.0]], np.float32)
    c2 = np.array([[0.1, 0.55, 0.0, 0.35], [0.0, 0.2, 0.8, 0.0]], np.float32)
    r = [np.array([0.3, 0.1], np.float32), np.array([0.5, 0.1], np.float32)]
    s = zr_score([c1, c2], r)
    # row 0: mean p = [.1, .5, 0, .4] -> exact, 1.0 * 0.5 + mean(0.3, 0.5)
    assert s[0] == pytest.approx(0.5 + 0.4, abs=1e-6)
    # row 1: irrelevant -> gain 0 -> only the regression mean
    assert s[1] == pytest.approx(0.1, abs=1e-6)
    assert zr_score([], r)[0] == pytest.approx(0.4, abs=1e-6)


def test_expected_gain_and_rank_norm():
    assert expected_gain([[0.45, 0.55, 0, 0]])[0] == pytest.approx(0.505)
    assert list(rank_norm([3.0, 1.0, 2.0, 9.0], [1, 1, 1, 2])) == pytest.approx([1.0, 1 / 3, 2 / 3, 1.0])


def test_fit_weights_finds_the_useful_member_and_prune_drops_noise():
    rng = np.random.default_rng(0)
    t = pd.DataFrame({"good": rng.random(200), "noise": rng.random(200)})
    target = t["good"].to_numpy()
    score = lambda s: -np.abs(s / max(s.max(), 1e-9) - target).mean()   # noqa: E731
    w, _ = fit_weights(t, ["good", "noise"], score)
    assert w["good"] > w["noise"]
    keep, _ = greedy_prune(["good", "noise"], lambda ms: score(weighted(t, ms, {m: 1.0 for m in ms})), {"noise": 5})
    assert keep == ["good"]
