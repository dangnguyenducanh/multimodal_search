"""Phase 4 cross-encoder (docs/4.md). Defaults reproduce ZhichunRoad's RobertaWithSampleWeight head:
[CLS] ++ brand emb(128) ++ color emb(128) -> dropout -> dense -> tanh -> dropout -> output.
Switches: 4-class / regression / both heads, multi-sample dropout, weighted layer pooling [www],
char 3-gram bags [ZR _Bag, hashed], image embedding block (Phase 5 variant C)."""
import torch
import torch.nn as nn
from transformers import AutoModel

BRAND_VOCAB, COLOR_VOCAB, BC_DIM = 151519, 47036, 128     # ZR's hard-coded table sizes
BAG_BUCKETS, BAG_DIM = 1 << 18, 64


class CrossEncoder(nn.Module):
    def __init__(self, backbone, head="reg", side=True, msd=False, wpool=False, bag=False, image_dim=0):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone, dtype=torch.float32)   # fp32 master weights; AMP casts
        h = self.enc.config.hidden_size
        p = getattr(self.enc.config, "hidden_dropout_prob", 0.1)
        self.head, self.side, self.msd, self.wpool, self.bag, self.image_dim = head, side, msd, wpool, bag, image_dim
        extra = 0
        if side:
            self.brand = nn.Embedding(BRAND_VOCAB, BC_DIM)
            self.color = nn.Embedding(COLOR_VOCAB, BC_DIM)
            extra += 2 * BC_DIM
        if bag:
            self.bags = nn.EmbeddingBag(BAG_BUCKETS, BAG_DIM, mode="sum")
            extra += 3 * BAG_DIM
        if image_dim:
            self.img = nn.Linear(image_dim, BC_DIM)
            self.no_img = nn.Parameter(torch.zeros(BC_DIM))
            extra += BC_DIM
        if wpool:
            self.layer_w = nn.Parameter(torch.zeros(4))
        self.drop = nn.Dropout(p)
        self.dense = nn.Linear(h + extra, h)
        self.out_reg = nn.Linear(h, 1) if head in ("reg", "both") else None
        self.out_cls = nn.Linear(h, 4) if head in ("cls", "both") else None
        std = self.enc.config.initializer_range                 # ZR's post_init: N(0, 0.02) for every new layer
        for m in [self.dense, self.out_reg, self.out_cls, getattr(self, "img", None)]:
            if m is not None:
                nn.init.normal_(m.weight, std=std)
                nn.init.zeros_(m.bias)
        for e in [getattr(self, n, None) for n in ("brand", "color", "bags")]:
            if e is not None:
                nn.init.normal_(e.weight, std=std)

    def forward(self, input_ids, attention_mask, brand_idx=None, color_idx=None, bag_ids=None, bag_offsets=None,
                img=None, has_img=None):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=self.wpool)
        if self.wpool:
            hs = torch.stack([x[:, 0] for x in out.hidden_states[-4:]], 0)
            x = (torch.softmax(self.layer_w, 0)[:, None, None] * hs).sum(0)
        else:
            x = out.last_hidden_state[:, 0]
        parts = [x]
        if self.side:
            parts += [self.brand(brand_idx), self.color(color_idx)]
        if self.bag:
            parts.append(self.bags(bag_ids, bag_offsets).view(x.size(0), 3 * BAG_DIM))
        if self.image_dim:
            e = self.img(img.to(x.dtype))
            parts.append(torch.where(has_img[:, None], e, self.no_img.to(e.dtype).expand_as(e)))
        z = torch.tanh(self.dense(self.drop(torch.cat(parts, 1))))
        drops = [self.drop(z) for _ in range(5 if (self.msd and self.training) else 1)]
        reg = torch.stack([self.out_reg(d) for d in drops]).mean(0).squeeze(-1) if self.out_reg is not None else None
        cls = torch.stack([self.out_cls(d) for d in drops]).mean(0) if self.out_cls is not None else None
        return reg, cls
