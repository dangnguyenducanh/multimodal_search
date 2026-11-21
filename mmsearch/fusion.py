"""Phase 5 variant A (docs/5.md): late fusion of a text reranker with CLIP image similarity.
Feature table -> GroupKFold(5) over queries -> combiner ladder A0 weighted sum, A1 logistic regression,
A2 LightGBM LambdaRank, A3 XGBoost ranker. Every reported number is out-of-fold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from mmsearch.esci import SQID
from mmsearch.images import image_table
from mmsearch.metrics import ndcg_per_query

LAB4 = {"E": 0, "S": 1, "C": 2, "I": 3}
RANK_LAB = {"I": 0, "C": 1, "S": 2, "E": 3}
GAIN = np.array([1.0, 0.1, 0.01, 0.0])


def _cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def feature_table(scores, queries):
    """scores: query_id, product_id, label, score [, pE..pI, reg]; queries: query_id -> query text."""
    df = scores.copy()
    has_p = "pE" in df and df["pE"].notna().all()
    df["eg"] = (df[["pE", "pS", "pC", "pI"]].to_numpy() @ GAIN) if has_p else df["score"]
    qf = pd.read_parquet(SQID / "query_features.parquet").set_index("query_id")["clip_text_features"]
    pt = pd.read_parquet(SQID / "product_features.parquet", columns=["product_id", "clip_text_features"]).set_index("product_id")["clip_text_features"]
    q = np.stack(qf.loc[df["query_id"]].to_numpy())
    it = image_table()["emb"]
    img = it.reindex(df["product_id"])
    df["has_img"] = img.notna().to_numpy()
    im = np.zeros_like(q)
    im[df["has_img"].to_numpy()] = np.stack(img.dropna().to_numpy())
    df["img_cos"] = np.where(df["has_img"], _cos(q, np.where(df["has_img"].to_numpy()[:, None], im, 1.0)), np.nan)
    txt = pt.reindex(df["product_id"])
    tv = np.stack([x if x is not None else np.ones(768) for x in txt.to_numpy()])
    df["txt_cos"] = _cos(q, tv)
    g = df.groupby("query_id")
    for c in ["score", "img_cos", "txt_cos"]:
        filled = df[c].fillna(g[c].transform("min"))           # a missing image sits at the bottom of its query
        df[f"{c}_f"] = filled.fillna(0)
        df[f"{c}_rank"] = g[c].rank(pct=True).fillna(0)
        df[f"{c}_gap"] = df[f"{c}_f"] - df.groupby("query_id")[f"{c}_f"].transform("max")
        sd = df.groupby("query_id")[f"{c}_f"].transform("std").replace(0, 1).fillna(1)
        df[f"{c}_z"] = (df[f"{c}_f"] - df.groupby("query_id")[f"{c}_f"].transform("mean")) / sd
    df["query_len"] = df["query_id"].map(queries).str.split().str.len()
    df["n_cand"] = g["product_id"].transform("size")
    return df


FEATURES = ["eg", "score_f", "has_img", "img_cos_f", "txt_cos_f"] + [f"{c}_{k}" for c in ["score", "img_cos", "txt_cos"]
                                                                       for k in ["rank", "gap", "z"]] + ["query_len", "n_cand"]


def _prob_cols(df):
    return [c for c in ["pE", "pS", "pC", "pI", "reg"] if c in df and df[c].notna().all()]


def fit_predict(kind, tr, te, seed=0):
    feats = FEATURES + _prob_cols(tr)
    if kind == "A0":                                       # one weight, picked on the training folds
        best = (-1, 0.0)
        for w in np.linspace(0, 0.5, 26):
            s = tr["score_z"] + w * tr["img_cos_z"]
            nd = ndcg_per_query(tr.assign(score=s)).mean()
            best = max(best, (nd, w))
        return (te["score_z"] + best[1] * te["img_cos_z"]).to_numpy(), {"w": best[1]}
    if kind == "A1":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(tr[feats])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(tr[feats]), tr["label"].map(LAB4))
        p = m.predict_proba(sc.transform(te[feats]))
        full = np.zeros((len(te), 4))
        full[:, m.classes_] = p
        return full @ GAIN, {}
    y_tr = tr["label"].map(RANK_LAB).to_numpy()
    groups = tr.groupby("query_id", sort=False).size().to_numpy()
    inner = tr["query_id"].isin(tr["query_id"].drop_duplicates().sample(frac=0.15, random_state=seed))
    fit, val = tr[~inner], tr[inner]
    if kind == "A2":
        import lightgbm as lgb
        m = lgb.LGBMRanker(objective="lambdarank", label_gain=[0, 1, 10, 100], num_leaves=15, min_data_in_leaf=50,
                           learning_rate=0.05, n_estimators=500, verbose=-1, random_state=seed)
        m.fit(fit[feats], fit["label"].map(RANK_LAB), group=fit.groupby("query_id", sort=False).size().to_numpy(),
              eval_set=[(val[feats], val["label"].map(RANK_LAB))],
              eval_group=[val.groupby("query_id", sort=False).size().to_numpy()],
              callbacks=[lgb.early_stopping(30, verbose=False)])
        return m.predict(te[feats]), {"importance": dict(zip(feats, m.booster_.feature_importance("gain").round(1)))}
    if kind == "A3":
        import xgboost as xgb
        m = xgb.XGBRanker(objective="rank:ndcg", ndcg_exp_gain=False, max_depth=4, eta=0.05, n_estimators=500,
                          early_stopping_rounds=30, random_state=seed)
        m.fit(fit[feats], fit["label"].map(RANK_LAB), qid=fit["query_id"], eval_set=[(val[feats], val["label"].map(RANK_LAB))],
              eval_qid=[val["query_id"]], verbose=False)
        return m.predict(te[feats]), {"importance": {k: round(float(v), 1) for k, v in m.get_booster().get_score(importance_type="total_gain").items()}}
    raise ValueError(kind)


def oof(df, kind, n=5):
    """Out-of-fold scores over queries (GroupKFold) + in-fold nDCG for the overfitting check."""
    df = df.sort_values(["query_id"]).reset_index(drop=True)
    out = np.zeros(len(df))
    infold, info = [], {}
    for k, (a, b) in enumerate(GroupKFold(n_splits=n).split(df, groups=df["query_id"])):
        tr, te = df.iloc[a], df.iloc[b]
        out[b], info = fit_predict(kind, tr, te, seed=k)
        s_tr, _ = fit_predict(kind, tr, tr, seed=k) if kind != "A0" else (None, None)
        if s_tr is not None:
            infold.append(ndcg_per_query(tr.assign(score=s_tr)).mean())
    return df.assign(score=out), {"infold_ndcg": float(np.mean(infold)) if infold else None, **info}
