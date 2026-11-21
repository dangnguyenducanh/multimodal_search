"""Phase 3 domain pretraining (docs/3.md), our reimplementation of [ZR]'s multi-task recipe:
masked words (15%, max 256) + Product2Query (a Poisson-length snippet of a product's text as a fake query,
2-class head, Poly-1 loss) + Product2Brand / Product2Color. Variants:
  zr     all three tasks (their recipe, minus the MoCo contrastive task, which gained nothing in their Table 3)
  cheap  masked words + Product2Query
  real   cheap, plus Product2Query positives/negatives from real train queries (Exact / Irrelevant products)
No test-product oversampling (their recipe was transductive). Output: runs/p3/<variant>/ (HF backbone)."""
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmsearch.esci import RUNS, load_p1_products, load_pairs
from mmsearch.text import product_fields, product_text


def snippet(words, rng, mean=4):
    n = max(1, min(len(words), rng.poisson(mean)))
    i = rng.integers(0, max(1, len(words) - n + 1))
    return " ".join(words[i:i + n])


def build_examples(variant, n_products, seed=0):
    rng = np.random.default_rng(seed)
    prod = load_p1_products()
    prod = prod.sample(min(n_products, len(prod)), random_state=seed).set_index("product_id")
    f = product_fields(prod, "www")
    text = product_text(f, "B", sep="</s></s>").to_numpy()
    ids = prod.index.to_numpy()
    title_words = [t.split() for t in f["title"]]
    q, d, y = [], [], []
    for i in range(len(ids)):
        s = snippet(title_words[i] or ["item"], rng)
        q += [s, s]
        d += [text[i], text[rng.integers(len(ids))]]
        y += [1, 0]
    ex = pd.DataFrame({"query": q, "doc": d, "p2q": y})
    if variant == "real":
        p = load_pairs(splits_=("train",))
        p = p[p["product_id"].isin(set(ids)) & p["label"].isin(["E", "I"])]
        pos = pd.Series(text, index=ids)
        ex = pd.concat([ex, pd.DataFrame({"query": p["query"].to_numpy(), "doc": pos.loc[p["product_id"]].to_numpy(),
                                          "p2q": (p["label"] == "E").astype(int).to_numpy()})], ignore_index=True)
    brands = f["brand"].str.lower().str.strip()
    colors = f["color"].str.lower().str.strip()
    bv = {b: i for i, b in enumerate(brands.value_counts()[lambda s: s >= 5].index)}
    cv = {c: i for i, c in enumerate(colors.value_counts()[lambda s: s >= 5].index)}
    doc_brand = dict(zip(text, [bv.get(b, -100) for b in brands]))
    doc_color = dict(zip(text, [cv.get(c, -100) for c in colors]))
    ex["brand"] = ex["doc"].map(doc_brand).fillna(-100).astype(int)
    ex["color"] = ex["doc"].map(doc_color).fillna(-100).astype(int)
    return ex.sample(frac=1, random_state=seed).reset_index(drop=True), len(bv), len(cv)


class MultiTask(nn.Module):
    def __init__(self, backbone, n_brand, n_color, tasks):
        super().__init__()
        from transformers import AutoModelForMaskedLM
        self.mlm = AutoModelForMaskedLM.from_pretrained(backbone, dtype=torch.float32)
        h = self.mlm.config.hidden_size
        self.p2q = nn.Linear(h, 2)
        self.tasks = tasks
        if "brandcolor" in tasks:
            self.brand, self.color = nn.Linear(h, n_brand), nn.Linear(h, n_color)

    def forward(self, input_ids, attention_mask, labels, p2q, brand, color):
        h = self.mlm.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        sel = labels != -100                            # vocab logits only at masked positions (250k-word vocab)
        mlm_loss = F.cross_entropy(self.mlm.lm_head(h[sel]).float(), labels[sel]) if sel.any() else h.sum() * 0
        cls = h[:, 0]
        logits = self.p2q(cls).float()
        pt = logits.softmax(-1).gather(1, p2q[:, None]).squeeze(1)
        loss = mlm_loss + (F.cross_entropy(logits, p2q, reduction="none") + (1 - pt)).mean()   # Poly-1, epsilon 1
        if "brandcolor" in self.tasks:
            loss = loss + 0.5 * (F.cross_entropy(self.brand(cls).float(), brand, ignore_index=-100)
                                 + F.cross_entropy(self.color(cls).float(), color, ignore_index=-100))
        return loss, logits


def run(variant="cheap", backbone="FacebookAI/xlm-roberta-base", n_products=200_000, max_len=256, batch=8,
        accum=16, lr=2e-5, epochs=1, seed=0):
    from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
    out = RUNS / "p3" / variant
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(backbone)
    ex, nb, nc = build_examples(variant, n_products, seed)
    held = ex.tail(2000)
    ex = ex.iloc[:-2000]
    tasks = {"zr": ["mlm", "p2q", "brandcolor"], "cheap": ["mlm", "p2q"], "real": ["mlm", "p2q"]}[variant]
    model = MultiTask(backbone, max(nb, 1), max(nc, 1), tasks).cuda()
    coll = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = math.ceil(len(ex) / batch / accum) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)

    def batches(df, shuffle):
        order = np.random.default_rng(seed).permutation(len(df)) if shuffle else np.arange(len(df))
        for i in range(0, len(df), batch):
            d = df.iloc[order[i:i + batch]]
            enc = tok(d["query"].tolist(), d["doc"].tolist(), truncation="only_second", max_length=max_len, padding=True)
            m = coll([{"input_ids": x} for x in enc["input_ids"]])
            yield {"input_ids": m["input_ids"].cuda(), "attention_mask": torch.tensor(enc["attention_mask"]).cuda(),
                   "labels": m["labels"].cuda(), "p2q": torch.tensor(d["p2q"].to_numpy()).cuda(),
                   "brand": torch.tensor(d["brand"].to_numpy()).cuda(), "color": torch.tensor(d["color"].to_numpy()).cuda()}

    def evaluate():
        model.eval()
        losses, acc = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in batches(held, False):
                loss, logits = model(**b)
                losses.append(loss.item())
                acc.append((logits.argmax(-1) == b["p2q"]).float().mean().item())
        model.train()
        return {"loss": float(np.mean(losses)), "p2q_acc": float(np.mean(acc))}

    before = evaluate()
    t0, step = time.time(), 0
    for _ in range(epochs):
        for i, b in enumerate(batches(ex, True)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(**b)
            (loss / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    print(f"[p3 {variant}] step {step}/{steps} loss {loss.item():.3f}", flush=True)
    after = evaluate()
    model.mlm.base_model.save_pretrained(out)          # the encoder only; heads are dropped for fine-tuning
    tok.save_pretrained(out)
    meta = {"variant": variant, "examples": len(ex), "gpu_hours": round((time.time() - t0) / 3600, 3),
            "held_out_before": before, "held_out_after": after}
    (out / "pretrain_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return out
