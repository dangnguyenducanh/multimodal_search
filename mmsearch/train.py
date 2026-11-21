"""Phase 4 trainer (docs/4.md). `Cfg()` defaults = ZhichunRoad's regression member at proxy scale ("ZR recipe @
proxy"): xlm-roberta-base, Task 1 US train rows, format B, 256 tokens padded, sigmoid-MSE on the gain, brand/color
side embeddings, FGM 0.5 + R-Drop 0.5 + EMA 0.999, lr 1e-5, batch 16 x 16, warmup 2%, 1 epoch, fp16.
Every ladder step is one field. Run: python -m mmsearch.train '{"name": "...", ...}'  (JSON overrides)
"""
import json
import math
import string
import sys
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from mmsearch.esci import RUNS, load_p1_products, load_pairs
from mmsearch.metrics import ndcg_per_query
from mmsearch.model import CrossEncoder
from mmsearch.text import budget_fields, mark_brands, product_fields, product_text

LABELS = np.array(["E", "S", "C", "I"])
GAINS = torch.tensor([1.0, 0.1, 0.01, 0.0])
ZR_VOCAB = RUNS / "p0" / "zr_data"


@dataclass
class Cfg:
    name: str = "zr-proxy"
    phase: str = "p4"
    backbone: str = "FacebookAI/xlm-roberta-base"
    data: str = "task1_us"          # task1_us | us | us_esjp
    clean: str = "zr"               # zr | www
    fmt: str = "B"                  # B | A
    order: str = "default"          # default | ptbdbc | ptdbbc | caption
    max_len: int = 256
    truncation: str = "longest_first"   # longest_first (ZR) | only_second
    dynamic_pad: bool = False       # ZR pads every pair to max_len
    budget: dict = None             # e.g. {"bullets": 96, "desc": 96}
    brand_markers: bool = False
    head: str = "reg"               # reg | cls | both
    cls_loss: str = "poly1"         # poly1 (ZR) | ce
    score: str = "auto"             # reg | zr | eg | sum ; auto: reg->reg, cls->zr, both->sum
    listwise: float = 0.0           # weight of a ListNet loss over each query's products (query-grouped batches)
    label_smoothing: float = 0.0
    fgm: float = 0.5
    awp: float = 0.0                # adversarial weight perturbation lr ([www]); replaces FGM when > 0
    rdrop: float = 0.5
    ema: float = 0.999
    msd: bool = False
    wpool: bool = False
    side: bool = True
    bag: bool = False
    image: bool = False
    lr: float = 1e-5
    batch: int = 16
    accum: int = 16
    warmup: float = 0.02
    epochs: float = 1.0
    precision: str = "fp16"
    rebalance: str = ""             # "" | zr | zr_title
    filter_drop: str = ""           # parquet (query_id, product_id) rows to drop (confident learning)
    distill: str = ""               # parquet of out-of-fold probs pE..pI: target = 0.7 onehot + 0.3 probs
    pseudo: str = ""                # parquet of pseudo-labelled pairs (query, product_id, pE..pI), weight 0.3
    oof: list = None                # [n_groups, k]: train on fold % n != k, predict fold % n == k
    eval_every: float = 0.5         # fraction of an epoch; the best dev checkpoint's scores are kept
    eval_splits: list = field(default_factory=lambda: ["dev"])
    seed: int = 12345789
    max_rows: int = 0               # smoke tests


def norm_key(s):                    # ZR's vocab key normalisation
    return str(s).translate(str.maketrans("\n", " ", string.punctuation)).lower().strip() if s else ""


def load_vocab(name):
    out = {}
    for i, line in enumerate((ZR_VOCAB / f"{name}.txt").read_text(encoding="utf8").split("\n")):
        if line and line != "[UNK]":
            out[line] = i
    return out


def char3(text, n=200):
    t = f"#{str(text).lower()[:n]}#"
    return [zlib.crc32(t[i:i + 3].encode()) % (1 << 18) for i in range(max(len(t) - 2, 1))]


class Data:
    """Tokenised rows + side inputs. Built once per (cfg text settings, rows)."""

    def __init__(self, cfg, rows, tok, products, fields_cache, image_store=None):
        self.n = len(rows)
        f = fields_cache.loc[rows["product_id"]].reset_index(drop=True)
        if cfg.budget:
            f = budget_fields(f, tok, cfg.budget)
        sep = f" {tok.sep_token}{tok.sep_token} " if tok.sep_token == "</s>" else f" {tok.sep_token} "
        text = product_text(f, cfg.fmt, cfg.order, sep=sep.strip())
        if "text_override" in rows:
            ov = rows["text_override"].to_numpy()
            text = pd.Series(np.where(pd.isna(ov), text.to_numpy(), ov))
        q = rows["query"].tolist()
        if cfg.brand_markers:
            q = mark_brands(q, brands_for_markers(products))
        enc = tok(q, text.tolist(), truncation=cfg.truncation, max_length=cfg.max_len,
                  padding=False if cfg.dynamic_pad else "max_length")
        self.ids, self.mask = enc["input_ids"], enc["attention_mask"]
        self.lens = np.array([len(x) for x in self.ids])
        self.label = rows["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy() if "label" in rows else None
        self.weight = rows["weight"].to_numpy(np.float32) if "weight" in rows else np.ones(self.n, np.float32)
        soft = rows[["pE", "pS", "pC", "pI"]].to_numpy(np.float32) if "pE" in rows else None
        self.soft = soft
        self.qid = rows["query_id"].to_numpy()
        if cfg.side:
            bv, cv = load_vocab("brand"), load_vocab("color")
            self.brand = np.array([bv.get(norm_key(b), 0) for b in f["brand"]])
            self.color = np.array([cv.get(norm_key(c), 0) for c in f["color"]])
        if cfg.bag:
            self.bag = [[char3(a), char3(b), char3(c)] for a, b, c in zip(q, f["title"], f["desc"])]
        if cfg.image:
            self.img, self.has_img = image_store(rows["product_id"])

    def batch(self, idx, cfg, dev):
        idx = np.asarray(idx)
        L = max(self.lens[idx]) if cfg.dynamic_pad else cfg.max_len
        ids = torch.full((len(idx), L), 1, dtype=torch.long)                 # pad id 1 for XLM-R; overwritten by mask
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            n = len(self.ids[i])
            ids[j, :n] = torch.tensor(self.ids[i])
            mask[j, :n] = torch.tensor(self.mask[i])
        b = {"input_ids": ids.to(dev), "attention_mask": mask.to(dev)}
        if cfg.side:
            b["brand_idx"] = torch.tensor(self.brand[idx]).to(dev)
            b["color_idx"] = torch.tensor(self.color[idx]).to(dev)
        if cfg.bag:
            flat, offs = [], []
            for i in idx:
                for g in self.bag[i]:
                    offs.append(len(flat))
                    flat += g
            b["bag_ids"], b["bag_offsets"] = torch.tensor(flat).to(dev), torch.tensor(offs).to(dev)
        if cfg.image:
            b["img"] = torch.tensor(self.img[idx]).to(dev)
            b["has_img"] = torch.tensor(self.has_img[idx]).to(dev)
        return b


_BRANDS = None


def brands_for_markers(products):
    global _BRANDS
    if _BRANDS is None:
        vc = products["product_brand"].dropna().str.strip().str.lower().value_counts()
        _BRANDS = [b for b in vc[vc >= 20].index if len(b) >= 3 and b.isascii()]
    return _BRANDS


def train_rows(cfg):
    locales = ("us", "es", "jp") if cfg.data.endswith("esjp") else ("us",)
    p = load_pairs(locales=locales, splits_=("train",))
    if cfg.data.startswith("task1"):
        p = p[p["small_version"] == 1]
    if cfg.oof:
        n, k = cfg.oof
        p = p[p["fold"] % n != k]
    extra = []
    if cfg.rebalance:
        from mmsearch.augment import rebalance
        f = product_fields(load_p1_products().set_index("product_id").loc[p["product_id"].unique()], cfg.clean)
        extra.append(rebalance(p, f, keep_title=cfg.rebalance == "zr_title"))
    if cfg.pseudo:
        ps = pd.read_parquet(cfg.pseudo)
        ps = ps.assign(weight=0.3, split="train")
        extra.append(ps)
    if cfg.filter_drop:
        d = pd.read_parquet(cfg.filter_drop)
        key = set(zip(d["query_id"], d["product_id"]))
        p = p[[k not in key for k in zip(p["query_id"], p["product_id"])]]
    p = pd.concat([p] + extra, ignore_index=True) if extra else p.reset_index(drop=True)
    if cfg.distill:
        oof = pd.read_parquet(cfg.distill)[["query_id", "product_id", "pE", "pS", "pC", "pI"]]
        p = p.merge(oof, on=["query_id", "product_id"], how="left")
        onehot = pd.get_dummies(p["label"]).reindex(columns=list(LABELS), fill_value=0).to_numpy(np.float32)
        probs = p[["pE", "pS", "pC", "pI"]].to_numpy(np.float32)
        probs = np.where(np.isnan(probs), onehot, probs)
        p[["pE", "pS", "pC", "pI"]] = 0.7 * onehot + 0.3 * probs
    if cfg.max_rows:
        p = p.sample(min(cfg.max_rows, len(p)), random_state=0)
    return p.reset_index(drop=True)


def eval_rows(split, cfg):
    if split == "oof":
        n, k = cfg.oof
        p = load_pairs(locales=("us",), splits_=("train",))
        return p[(p["fold"] % n == k) & ((p["small_version"] == 1) | ~cfg.data.startswith("task1"))].reset_index(drop=True)
    p = load_pairs(splits_=(split,))
    if cfg.max_rows:
        p = p[p["query_id"].isin(p["query_id"].drop_duplicates().head(50))]
    return p.reset_index(drop=True)


def batches(data, cfg, rng):
    idx = rng.permutation(data.n)
    if cfg.listwise > 0:                         # whole queries per batch (a long query is split)
        groups = pd.Series(np.arange(data.n)).groupby(data.qid).apply(list).tolist()
        rng.shuffle(groups)
        out, cur = [], []
        for g in groups:
            for i in range(0, len(g), cfg.batch * 2):
                part = g[i:i + cfg.batch * 2]
                if cur and len(cur) + len(part) > cfg.batch * 2:
                    out.append(cur)
                    cur = []
                cur += part
        return out + ([cur] if cur else [])
    bs = [idx[i:i + cfg.batch] for i in range(0, data.n, cfg.batch)]
    if cfg.dynamic_pad:                          # length-sorted inside chunks of 50 batches, then shuffled
        chunks = [idx[i:i + cfg.batch * 50] for i in range(0, data.n, cfg.batch * 50)]
        bs = []
        for c in chunks:
            c = c[np.argsort(data.lens[c])]
            bs += [c[i:i + cfg.batch] for i in range(0, len(c), cfg.batch)]
        rng.shuffle(bs)
    return bs


def score_of(reg, cls, cfg):
    mode = cfg.score if cfg.score != "auto" else {"reg": "reg", "cls": "zr", "both": "sum"}[cfg.head]
    eg = (cls.softmax(-1) * GAINS.to(cls.device)).sum(-1) if cls is not None else None
    if mode == "reg":
        return torch.sigmoid(reg)
    if mode == "eg":
        return eg
    if mode == "zr":
        p = cls.softmax(-1)
        top = p.argmax(-1)
        return GAINS.to(p.device)[top] * p.max(-1).values
    return eg + torch.sigmoid(reg)               # sum


def loss_fn(reg, cls, b, cfg, dev):
    w = b["w"]
    total = 0.0
    if reg is not None:
        target = b["gain"]
        total = total + (w * (torch.sigmoid(reg.float()) - target) ** 2).sum() / w.sum()
    if cls is not None:
        logp = F.log_softmax(cls.float(), -1)
        t = b["soft"] if b.get("soft") is not None else F.one_hot(b["y"], 4).float()
        if cfg.label_smoothing:
            t = (1 - cfg.label_smoothing) * t + cfg.label_smoothing / 4
        ce = -(t * logp).sum(-1)
        if cfg.cls_loss == "poly1":
            ce = ce + (1 - (t * logp.exp()).sum(-1))
        total = total + (w * ce).sum() / w.sum()
    if cfg.listwise > 0:
        s = reg.float() if reg is not None else 10 * (cls.float().softmax(-1) * GAINS.to(dev)).sum(-1)
        q = b["q"]
        lw = 0.0
        groups = 0
        for u in torch.unique(q):
            m = q == u
            g = b["gain"][m]
            if g.sum() <= 0 or m.sum() < 2:
                continue
            lw = lw - ((g / g.sum()) * F.log_softmax(s[m], 0)).sum()
            groups += 1
        if groups:
            total = total + cfg.listwise * lw / groups
    return total


def rdrop_kl(a, b):                              # ZR util_loss_fct 'r-drop-kl': summed symmetric KL
    a, b = a.float(), b.float()
    return (F.kl_div(F.log_softmax(a, -1), F.softmax(b, -1), reduction="sum")
            + F.kl_div(F.log_softmax(b, -1), F.softmax(a, -1), reduction="sum"))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                tmp = p.detach().clone()
                p.copy_(self.shadow[n].to(p.dtype))
                self.shadow[n] = tmp.float()


def adversarial_step(model, b, cfg, scaler, reg, cls, running_loss):
    """FGM on word embeddings [ZR] or AWP on weights [www]; returns adversarial logits for R-Drop."""
    if cfg.awp > 0 and running_loss < (0.6 if cfg.head != "reg" else 1.0):
        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None
                  and "weight" in n and p.dim() > 1 and "embeddings" not in n]
        backup = {}
        for n, p in params:
            backup[n] = p.data.clone()
            g = p.grad.float()
            r = cfg.awp * g / (g.norm() + 1e-6) * (p.data.float().norm() + 1e-6)
            p.data.add_(r.to(p.dtype))
            eps = 1e-2 * backup[n].abs()
            p.data.copy_(torch.max(torch.min(p.data, backup[n] + eps), backup[n] - eps))
        out = run_loss(model, b, cfg, scaler)
        for n, p in params:
            p.data.copy_(backup[n])
        return out
    if cfg.fgm > 0:
        emb = model.enc.get_input_embeddings().weight
        if emb.grad is None:
            return None
        g = emb.grad.float()
        norm = g.norm()
        if norm == 0 or torch.isnan(norm):
            return None
        backup = emb.data.clone()
        emb.data.add_((cfg.fgm * g / norm).to(emb.dtype))
        out = run_loss(model, b, cfg, scaler)
        emb.data.copy_(backup)
        return out
    if cfg.rdrop > 0 and cls is not None:        # plain R-Drop: a second dropout pass
        return run_loss(model, b, cfg, scaler)
    return None


def run_loss(model, b, cfg, scaler, backward=True):
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg.precision)
    with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
        reg, cls = model(**b["x"])
        loss = loss_fn(reg, cls, b, cfg, b["x"]["input_ids"].device) / cfg.accum
    if backward:
        scaler.scale(loss).backward(retain_graph=cfg.rdrop > 0 and cls is not None)
    return reg, cls, loss


@torch.no_grad()
def predict(model, data, cfg, dev):
    model.eval()
    out_reg, out_p = [], []
    order = np.argsort(data.lens)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg.precision)
    for i in range(0, data.n, 128):
        idx = order[i:i + 128]
        with torch.autocast("cuda", dtype=dt, enabled=dt is not None):
            reg, cls = model(**data.batch(idx, cfg, dev))
        out_reg.append(torch.sigmoid(reg.float()).cpu().numpy() if reg is not None else np.full(len(idx), np.nan))
        out_p.append(cls.float().softmax(-1).cpu().numpy() if cls is not None else np.full((len(idx), 4), np.nan))
    inv = np.argsort(order)
    reg, p = np.concatenate(out_reg)[inv], np.concatenate(out_p)[inv]
    model.train()
    return reg, p


def to_scores(rows, reg, p, cfg):
    df = rows[["query_id", "product_id", "label"]].copy()
    df["reg"] = reg
    df[["pE", "pS", "pC", "pI"]] = p
    t_reg = torch.tensor(reg) if not np.isnan(reg).all() else None
    t_cls = torch.tensor(np.log(np.clip(p, 1e-9, 1))) if not np.isnan(p).all() else None
    if t_reg is not None:
        t_reg = torch.logit(t_reg.clamp(1e-7, 1 - 1e-7))
    df["score"] = score_of(t_reg, t_cls, cfg).numpy()
    return df


def run(cfg, write=True):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from mmsearch.runlog import write_run
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = torch.device("cuda")
    out = RUNS / cfg.phase / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(json.dumps(asdict(cfg), indent=1))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(cfg.backbone)
    products = load_p1_products().set_index("product_id")
    rows = train_rows(cfg)
    need = pd.Index(rows["product_id"].unique())
    evals = {s: eval_rows(s, cfg) for s in cfg.eval_splits + (["oof"] if cfg.oof else [])}
    for e in evals.values():
        need = need.union(pd.Index(e["product_id"].unique()))
    pr = products.loc[need]
    if cfg.order == "caption":
        caps = pd.read_parquet(RUNS / "p5" / "captions.parquet").set_index("product_id")["caption"]
        pr = pr.assign(caption=caps.reindex(pr.index).fillna(""))
    fields = product_fields(pr, cfg.clean)
    store = None
    if cfg.image:
        from mmsearch.images import image_store
        store = image_store()
    train = Data(cfg, rows, tok, products, fields, store)
    train.gain = rows["label"].map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}).to_numpy(np.float32)
    if train.soft is not None:                    # soft targets also define the regression target
        soft = np.nan_to_num(train.soft, nan=-1)
        has = soft[:, 0] >= 0
        train.gain = np.where(has, soft @ GAINS.numpy(), train.gain)
    ev = {s: Data(cfg, r, tok, products, fields, store) for s, r in evals.items()}
    t_data = time.time() - t0

    model = CrossEncoder(cfg.backbone, cfg.head, cfg.side, cfg.msd, cfg.wpool, cfg.bag,
                         768 if cfg.image else 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    n_batches = math.ceil(train.n / cfg.batch) if cfg.listwise == 0 else len(batches(train, cfg, np.random.default_rng(0)))
    steps = math.ceil(n_batches * cfg.epochs / cfg.accum)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
    ema = EMA(model, cfg.ema) if cfg.ema else None
    dev_rows = evals.get("dev")
    best = {"ndcg": -1}
    evals_at = {int(round(f * steps)) for f in np.arange(cfg.eval_every, cfg.epochs + 1e-9, cfg.eval_every)} | {steps}
    step, micro, running = 0, 0, 1.0
    t_train = time.time()
    model.train()
    while step < steps:
        for idx in batches(train, cfg, rng):
            b = {"x": train.batch(idx, cfg, dev), "w": torch.tensor(train.weight[idx]).to(dev),
                 "gain": torch.tensor(train.gain[idx]).to(dev), "y": torch.tensor(train.label[idx]).to(dev),
                 "q": torch.tensor(train.qid[idx]).to(dev),
                 "soft": torch.tensor(train.soft[idx]).to(dev) if train.soft is not None and
                 not np.isnan(train.soft[idx]).any() else None}
            reg, cls, loss = run_loss(model, b, cfg, scaler)
            running = 0.98 * running + 0.02 * loss.item() * cfg.accum
            adv = adversarial_step(model, b, cfg, scaler, reg, cls, running)
            if adv is not None and cfg.rdrop > 0 and cls is not None:
                kl = cfg.rdrop * rdrop_kl(adv[1], cls) / cfg.accum
                scaler.scale(kl).backward()
            micro += 1
            if micro % cfg.accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if ema:
                    ema.update(model)
                if step % 50 == 0:
                    print(f"[{cfg.name}] step {step}/{steps} loss {running:.4f} "
                          f"{(time.time() - t_train) / step:.2f}s/step", flush=True)
                if step in evals_at and dev_rows is not None:
                    if ema:
                        ema.swap(model)
                    reg_d, p_d = predict(model, ev["dev"], cfg, dev)
                    s = to_scores(dev_rows, reg_d, p_d, cfg)
                    nd = float(ndcg_per_query(s).mean())
                    print(f"[{cfg.name}] eval step {step}: dev nDCG {nd:.4f}", flush=True)
                    if nd > best["ndcg"]:
                        best = {"ndcg": nd, "step": step}
                        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "model.pt")
                    if ema:
                        ema.swap(model)
                if step >= steps:
                    break
    gpu_h = (time.time() - t_train) / 3600
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    results, ms = {}, None
    for s, d in ev.items():
        t1 = time.time()
        reg_e, p_e = predict(model, d, cfg, dev)
        if s == "dev":
            ms = 1000 * (time.time() - t1) / evals["dev"]["query_id"].nunique()
        sc = to_scores(evals[s], reg_e, p_e, cfg)
        sc.to_parquet(out / f"scores_{s}.parquet", index=False)
        results[s] = sc
    meta = {"best": best, "gpu_hours": round(gpu_h, 3), "data_s": round(t_data), "rows": train.n, "steps": steps}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    if write and "dev" in results:
        splits = pd.read_parquet(RUNS / "p0" / "splits.parquet")["split"]
        slices = pd.read_parquet(RUNS / "p0" / "slices.parquet")
        pq = pd.concat([ndcg_per_query(results[s]) for s in results if s in ("dev", "final_test")])
        if cfg.max_rows:
            print(f"[{cfg.name}] smoke dev nDCG {pq.mean():.4f} (50 queries)")
        else:
            write_run(cfg.phase, cfg.name, pq, splits, slices, report_test="final_test" in results,
                      args=asdict(cfg), models=[str(out / "model.pt")], gpu_hours=round(gpu_h, 3),
                      ms_per_query=round(ms, 1) if ms else None, notes=cfg.name)
    return results


def predict_run(phase, name, split="final_test", rows=None):
    """Reload a finished run (cfg.json + model.pt) and score another split (or given rows: query, product_id, ...)."""
    from transformers import AutoTokenizer
    out = RUNS / phase / name
    dst = out / f"scores_{split}.parquet"
    if rows is None and dst.exists():
        return pd.read_parquet(dst)
    cfg = Cfg(**json.loads((out / "cfg.json").read_text()))
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(cfg.backbone)
    products = load_p1_products().set_index("product_id")
    r = rows if rows is not None else eval_rows(split, cfg)
    if "label" not in r:
        r = r.assign(label="I")
    pr = products.loc[pd.Index(r["product_id"].unique())]
    if cfg.order == "caption":
        caps = pd.read_parquet(RUNS / "p5" / "captions.parquet").set_index("product_id")["caption"]
        pr = pr.assign(caption=caps.reindex(pr.index).fillna(""))
    store = None
    if cfg.image:
        from mmsearch.images import image_store
        store = image_store()
    d = Data(cfg, r.reset_index(drop=True), tok, products, product_fields(pr, cfg.clean), store)
    model = CrossEncoder(cfg.backbone, cfg.head, cfg.side, cfg.msd, cfg.wpool, cfg.bag, 768 if cfg.image else 0).to(dev)
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    reg, p = predict(model, d, cfg, dev)
    sc = to_scores(r.reset_index(drop=True), reg, p, cfg)
    if rows is None:
        sc.to_parquet(dst, index=False)
    return sc


if __name__ == "__main__":
    run(Cfg(**json.loads(sys.argv[1] if len(sys.argv) > 1 else "{}")))
