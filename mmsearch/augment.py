"""Phase 2 augmentations (docs/2.md), each a switch: [ZR] translation, C/I rebalancing, confident-learning filter.
Augmented rows get weight 0.3 ([ZR] sample-weight rule)."""
import numpy as np
import pandas as pd

AUG_WEIGHT = 0.3


def rebalance(pairs, fields, keep_title=False, max_chars=800):
    """[ZR] 6_rebalance_negative_samples.py: extra Complement/Irrelevant train pairs whose product text is
    bullets + description (cut to ~800 chars). keep_title=True is our variant (docs/2.md proposal 9)."""
    ci = pairs[(pairs["split"] == "train") & pairs["label"].isin(["C", "I"])].copy()
    f = fields.loc[ci["product_id"]]
    body = (f["bullets"].to_numpy() + " " + f["desc"].to_numpy())
    body = pd.Series(body).str.slice(0, max_chars).to_numpy()
    ci["text_override"] = (f["title"].to_numpy() + " " + body) if keep_title else body
    ci["weight"] = AUG_WEIGHT
    ci["augmented"] = "rebalance"
    return ci[ci["text_override"].str.strip().str.len() > 0]


def translate(texts, src, batch=64, device="cuda"):
    """[ZR] 4_translation_es_us.py: MarianMT opus-mt-{es,ja}-en, English as the pivot."""
    from transformers import MarianMTModel, MarianTokenizer
    name = {"es": "Helsinki-NLP/opus-mt-es-en", "jp": "Helsinki-NLP/opus-mt-ja-en"}[src]
    tok, model = MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name).to(device).half().eval()
    out = []
    import torch
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=128, num_beams=1)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return out


def confident_filter(labels, oof_probs, frac=0.04):
    """[ZR] 7_confidient_learning_filter.py: cleanlab on out-of-fold class probabilities; returns a boolean mask
    of rows to DROP (the top `frac` label issues ranked by normalized margin)."""
    from cleanlab.filter import find_label_issues
    idx = find_label_issues(labels=np.asarray(labels), pred_probs=np.asarray(oof_probs),
                            filter_by="prune_by_noise_rate", return_indices_ranked_by="normalized_margin")
    drop = np.zeros(len(labels), dtype=bool)
    drop[idx[: int(frac * len(labels))]] = True
    return drop
