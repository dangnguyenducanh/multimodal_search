"""Ensemble scoring (docs/6.md). Phase 0 needs only zr_score(); Phase 6 adds the improvements."""
import numpy as np

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
