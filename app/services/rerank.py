"""Rerankers for the demo (docs/9.md, docs/demo-handoff.md).

precomputed  any host (Vercel too): for a known dev/test query, its judged products ranked by a model's saved
             scores (table pair_scores, filled by scripts/p9_add_models.py). Free-text queries stay BM25-only.
live         GPU host only: MMS_RERANKER (text) / MMS_MM_RERANKER (multimodal) = runs/<phase>/<run>; the model is
             loaded once at startup and reranks BM25's top 100 for any query (imports mmsearch + torch lazily).
A reranker r is called r(db, query, hits, query_id) -> hits, or None when it can't rank this query;
r.model(db) names the model it ranks with (None when there is none)."""
import os
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

from app.db import has_table
from app.schemas import Product, ProductHit, Probs

LABELS = ("E", "S", "C", "I")
# models.role per search mode. Multimodal falls back to the text winner, so the default mode is never blank.
ROLES = {"text": ("ours",), "multimodal": ("multimodal", "ours")}


def model_for(db: sqlite3.Connection, *roles: str) -> str | None:
    if not has_table(db, "pair_scores"):
        return None
    for role in roles:
        r = db.execute("SELECT name FROM models WHERE role = ? ORDER BY dev_ndcg DESC LIMIT 1", (role,)).fetchone()
        if r:
            return r[0]
    return None


def probs_of(pE, pS, pC, pI) -> tuple[Probs | None, str | None]:
    if pE is None or pE != pE:  # SQL NULL / NaN: a regression-only model
        return None, None
    p = Probs(E=float(pE), S=float(pS), C=float(pC), I=float(pI))
    return p, max(LABELS, key=lambda k: getattr(p, k))


class Precomputed:
    def __init__(self, roles: tuple[str, ...]):
        self.roles = roles

    def model(self, db):
        return model_for(db, *self.roles)

    def __call__(self, db, q, hits, qid):
        name = self.model(db)
        if qid is None or name is None:
            return None
        rows = db.execute("""SELECT p.*, s.score, s.pE, s.pS, s.pC, s.pI FROM pair_scores s
                             JOIN products p ON p.product_id = s.product_id
                             WHERE s.query_id = ? AND s.model = ? ORDER BY s.score DESC, p.product_id""",
                          (qid, name)).fetchall()
        out = []
        for r in rows:
            probs, pred = probs_of(r["pE"], r["pS"], r["pC"], r["pI"])
            out.append(ProductHit(**{k: r[k] for k in Product.model_fields}, score=round(r["score"], 4),
                                  probs=probs, predicted=pred))
        return out or None


class Live:
    """Our trained cross-encoder, loaded once (GPU host)."""

    def __init__(self, run_dir: str):
        # Project rule: all our GPU processes together stay under 25 GB. Importing mmsearch caps this process at
        # MMS_GPU_GB; the queue (scripts/run_all.sh) takes 24, so run Live while it is idle or lower both.
        os.environ.setdefault("MMS_GPU_GB", "8")
        import json

        import pandas as pd
        import torch
        from transformers import AutoTokenizer

        from mmsearch import cap_gpu
        from mmsearch.esci import RUNS
        from mmsearch.model import CrossEncoder
        from mmsearch.train import Cfg, product_table
        cap_gpu()
        run = Path(run_dir)
        self.name = "/".join(run.resolve().parts[-2:])
        self.cfg = c = Cfg(**json.loads((run / "cfg.json").read_text()))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu = self.dev.type == "cuda"
        self.tok = AutoTokenizer.from_pretrained(c.backbone)
        pt = product_table()                       # keyed product_id|locale, like the trainer
        self.products = pt[pt["product_locale"] == "us"]
        if c.order == "caption":  # Phase 5 variant B reads the caption field (same as train.predict_run)
            caps = pd.read_parquet(RUNS / "p5" / "captions.parquet").set_index("product_id")["caption"]
            self.products = self.products.assign(caption=caps.reindex(self.products["product_id"]).fillna("").to_numpy())
        self.net = CrossEncoder(c.backbone, c.head, c.side, c.msd, c.wpool, c.bag, 768 if c.image else 0).to(self.dev)
        self.net.load_state_dict(torch.load(run / "model.pt", map_location=self.dev))
        self.net.eval()
        self.store = None
        if c.image:
            from mmsearch.images import image_store
            self.store = image_store()
        # ponytail: one request at a time (train.predict flips model.train() at its end); a worker pool if traffic.
        self.lock = threading.Lock()

    def model(self, db):
        return self.name

    def score(self, query: str, product_ids: list[str]):
        import pandas as pd

        from mmsearch.text import product_fields
        from mmsearch.train import Data, pkeys, predict, to_scores
        rows = pd.DataFrame({"query_id": 0, "query": query, "product_id": product_ids, "label": "I", "locale": "us"})
        fields = product_fields(self.products.loc[pkeys(rows).unique()], self.cfg.clean)
        d = Data(self.cfg, rows, self.tok, self.products, fields, self.store)
        with self.lock:
            reg, p = predict(self.net, d, self.cfg, self.dev)
        return to_scores(rows, reg, p, self.cfg)

    def __call__(self, db, q, hits, qid):
        if not hits:
            return None
        s = self.score(q, [h.product_id for h in hits]).set_index("product_id")
        raw, out = {}, []
        for h in hits:
            r = s.loc[h.product_id]
            raw[h.product_id] = float(r["score"])
            probs, pred = probs_of(r["pE"], r["pS"], r["pC"], r["pI"])
            out.append(h.model_copy(update={"score": round(raw[h.product_id], 4), "probs": probs, "predicted": pred}))
        return sorted(out, key=lambda h: (-raw[h.product_id], h.product_id))  # unrounded, as the offline harness


@lru_cache(maxsize=2)
def live(run_dir: str) -> Live:
    return Live(run_dir)


def install(rerankers: dict) -> None:
    """Live models when configured (GPU host), otherwise the precomputed tables."""
    text, mm = os.environ.get("MMS_RERANKER"), os.environ.get("MMS_MM_RERANKER")
    rerankers["text"] = live(text) if text else Precomputed(ROLES["text"])
    rerankers["multimodal"] = live(mm or text) if mm or text else Precomputed(ROLES["multimodal"])
