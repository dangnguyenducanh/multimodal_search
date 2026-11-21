"""Phase 8 item-to-item recommendations (docs/8.md): the viewed product's title is the query, the reranker's
class probabilities rank the candidates (Alternatives = P(S), Goes with = P(C)), optionally + image similarity."""
import numpy as np
import pandas as pd

from mmsearch.text import clean


def seed_rows(seeds, cands, products):
    """(seed product, candidate) pairs as reranker rows: query = the seed's cleaned title."""
    titles = products.loc[seeds, "product_title"].map(lambda s: clean(s, "www")).to_numpy()
    return pd.DataFrame({"query_id": np.arange(len(seeds)), "seed": seeds, "query": titles, "product_id": cands})


def image_sim(seeds, cands, store):
    a, ha = store(seeds)
    b, hb = store(cands)
    s = (a * b).sum(1)
    return np.where(ha & hb, s, np.nan)


def dedupe(seed_title, cand_titles):
    """Drop the viewed product's near-duplicates (same title after cleaning)."""
    t = clean(seed_title, "www").lower()
    return [clean(c, "www").lower() != t for c in cand_titles]
