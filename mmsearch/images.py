"""Phase 5 / 7 image side (docs/5.md, docs/7.md): CLIP-L/14 embeddings (SQID's model) and the product image store."""
from functools import lru_cache

import numpy as np
import pandas as pd

from mmsearch.esci import ROOT, RUNS, SQID

CLIP = "openai/clip-vit-large-patch14"
IMAGES = ROOT / "data" / "images"


@lru_cache(maxsize=1)
def _clip(device="cuda"):
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP, dtype=torch.float16).to(device).eval()
    return model, CLIPProcessor.from_pretrained(CLIP)


def _feat(r):
    """transformers 5 returns an output object from get_*_features; older versions return the tensor."""
    import torch
    t = r if torch.is_tensor(r) else r.pooler_output
    assert t.shape[-1] == 768, t.shape            # the projected CLIP space, not the 1024-d vision hidden state
    return t.float().cpu().numpy()


def encode_images(paths, batch=32, device="cuda"):
    """Unnormalised CLIP image embeddings (SQID stores them unnormalised too). Unreadable files -> NaN rows."""
    import torch
    from PIL import Image
    model, proc = _clip(device)
    out = np.full((len(paths), 768), np.nan, dtype=np.float32)
    for i in range(0, len(paths), batch):
        imgs, ok = [], []
        for j, p in enumerate(paths[i:i + batch]):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                ok.append(i + j)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device, torch.float16)
            out[ok] = _feat(model.get_image_features(pixel_values=px))
    return out


def encode_texts(texts, batch=256, device="cuda"):
    import torch
    model, proc = _clip(device)
    out = []
    for i in range(0, len(texts), batch):
        with torch.no_grad():
            t = proc(text=list(texts[i:i + batch]), return_tensors="pt", padding=True, truncation=True, max_length=77)
            out.append(_feat(model.get_text_features(**{k: v.to(device) for k, v in t.items()})))
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


@lru_cache(maxsize=1)
def image_table():
    """product_id -> CLIP image embedding: SQID's (test products) + ours (supplementary products)."""
    sq = pd.read_parquet(SQID / "product_features.parquet", columns=["product_id", "clip_image_features"]).dropna()
    parts = [sq.rename(columns={"clip_image_features": "emb"})]
    extra = RUNS / "p5" / "image_features_extra.parquet"
    if extra.exists():
        parts.append(pd.read_parquet(extra))
    t = pd.concat(parts, ignore_index=True).drop_duplicates("product_id").set_index("product_id")
    return t


def image_store():
    """For the trainer's image block: returns f(product_ids) -> (emb [n,768] float32, has_img [n] bool)."""
    t = image_table()
    mat = np.stack(t["emb"].to_numpy()).astype(np.float32)
    pos = pd.Series(np.arange(len(t)), index=t.index)

    def get(pids):
        idx = pos.reindex(pd.Index(pids)).to_numpy()
        has = ~np.isnan(idx)
        emb = np.zeros((len(idx), 768), np.float32)
        emb[has] = mat[idx[has].astype(int)]
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-6)
        return emb, has
    return get
