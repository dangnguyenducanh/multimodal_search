"""Phase 5 inputs: (1) CLIP-L/14 image embeddings for imaged products SQID has no features for, checked against
SQID's own vectors on test products; (2) one caption per image (variant B).
  python scripts/p5_image_features.py --clip      -> runs/p5/image_features_extra.parquet, clip_check.json
  python scripts/p5_image_features.py --captions  -> runs/p5/captions.parquet  (resumable)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmsearch.esci import RUNS, SQID  # noqa: E402
from mmsearch.images import IMAGES, encode_images  # noqa: E402

CAPTIONER = "Salesforce/blip-image-captioning-base"
OUT = RUNS / "p5"


def clip_features():
    sq = pd.read_parquet(SQID / "product_features.parquet", columns=["product_id", "clip_image_features"]).dropna()
    # check: same model + preprocessing as SQID -> cosine ~ 1 on their products
    sample = sq.sample(200, random_state=0)
    sample = sample[[(IMAGES / f"{p}.jpg").exists() for p in sample["product_id"]]]
    ours = encode_images([IMAGES / f"{p}.jpg" for p in sample["product_id"]])
    theirs = np.stack(sample["clip_image_features"].to_numpy())
    cos = (ours * theirs).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1))
    check = {"n": len(sample), "cos_mean": float(np.nanmean(cos)), "cos_p05": float(np.nanpercentile(cos, 5)),
             "norm_ratio": float(np.nanmean(np.linalg.norm(ours, axis=1) / np.linalg.norm(theirs, axis=1)))}
    print("CLIP check vs SQID:", check, flush=True)
    (OUT / "clip_check.json").write_text(json.dumps(check, indent=1))
    have = set(sq["product_id"])
    todo = [p.stem for p in IMAGES.glob("*.jpg") if p.stem not in have]
    emb = encode_images([IMAGES / f"{p}.jpg" for p in todo])
    keep = ~np.isnan(emb).any(1)
    pd.DataFrame({"product_id": np.array(todo)[keep], "emb": list(emb[keep])}).to_parquet(
        OUT / "image_features_extra.parquet", index=False)
    print(f"extra image features: {keep.sum():,} of {len(todo):,}", flush=True)


def captions(batch=48):
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor
    dst = OUT / "captions.parquet"
    done = pd.read_parquet(dst) if dst.exists() else pd.DataFrame(columns=["product_id", "caption"])
    seen = set(done["product_id"])
    todo = sorted(p.stem for p in IMAGES.glob("*.jpg") if p.stem not in seen)
    proc = BlipProcessor.from_pretrained(CAPTIONER)
    model = BlipForConditionalGeneration.from_pretrained(CAPTIONER, dtype=torch.float16).to("cuda").eval()
    rows = []
    for i in range(0, len(todo), batch):
        pids, imgs = [], []
        for p in todo[i:i + batch]:
            try:
                imgs.append(Image.open(IMAGES / f"{p}.jpg").convert("RGB"))
                pids.append(p)
            except Exception:
                pass
        with torch.no_grad():
            x = proc(images=imgs, return_tensors="pt").to("cuda", torch.float16)
            out = model.generate(**x, max_new_tokens=30)
        rows += list(zip(pids, proc.batch_decode(out, skip_special_tokens=True)))
        if (i // batch) % 200 == 0 or i + batch >= len(todo):
            done = pd.concat([done, pd.DataFrame(rows, columns=["product_id", "caption"])], ignore_index=True)
            done.to_parquet(dst, index=False)
            rows = []
            print(f"captions {len(done):,}/{len(done) + len(todo) - i - batch:,}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="store_true")
    ap.add_argument("--captions", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.clip:
        clip_features()
    if a.captions:
        captions()
