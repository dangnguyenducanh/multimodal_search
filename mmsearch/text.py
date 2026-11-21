"""Phase 2 (docs/2.md): cleaning and product-text formats. Each option is a switch the proxy ladder tests.

Formats:  B  = [ZR] `title SEP brand SEP color SEP bullets SEP desc` (SEP = the tokenizer's own separator pair)
          A  = [www] `color: c brand: b description: title bullets desc`
Orders (B only, [ZR] ensemble diversity): default / ptbdbc (title bullets desc brand color) / ptdbbc (title desc bullets brand color)
Cleaning: "zr"  = strip HTML from the description only, empty -> ''          ([ZR] preprocess_function)
          "www" = unescape + strip HTML, drop symbols/emojis, collapse whitespace, on every field
"""
import html
import re

import pandas as pd

FIELDS = {"title": "product_title", "brand": "product_brand", "color": "product_color",
          "bullets": "product_bullet_point", "desc": "product_description"}
ORDERS = {"default": ["title", "brand", "color", "bullets", "desc"],
          "ptbdbc": ["title", "bullets", "desc", "brand", "color"],
          "ptdbbc": ["title", "desc", "bullets", "brand", "color"],
          "caption": ["title", "brand", "color", "caption", "bullets", "desc"]}     # Phase 5 variant B
_TAG = re.compile(r"<.*?>")                      # ZR's regex
_SYMBOLS = re.compile(r"[^\w\s.,;:!?'\"()&/%+#-]", flags=re.UNICODE)
_WS = re.compile(r"\s+")


def _blank(s):
    return "" if s is None or (isinstance(s, float) and s != s) or str(s).isspace() else str(s)


def clean(s, how="zr", field=""):
    s = _blank(s)
    if not s:
        return ""
    if how == "zr":
        return _TAG.sub("", s) if field == "desc" else s
    s = _TAG.sub(" ", html.unescape(s))
    return _WS.sub(" ", _SYMBOLS.sub(" ", s)).strip()


def product_fields(products, how="zr", max_chars=4000):
    """products DataFrame (ESCI columns) -> DataFrame of cleaned fields title/brand/color/bullets/desc[/caption]."""
    out = pd.DataFrame(index=products.index)
    for f, col in FIELDS.items():
        out[f] = [clean(v, how, f)[:max_chars] for v in products[col].tolist()]
    if "caption" in products:
        out["caption"] = products["caption"].fillna("").astype(str)
    return out


def product_text(fields, fmt="B", order="default", sep="</s></s>"):
    """Cleaned fields -> one product string per row."""
    if fmt == "A":
        img = ("image: " + fields["caption"] + " ") if order == "caption" and "caption" in fields else ""
        return ("color: " + fields["color"] + " brand: " + fields["brand"] + " " + img + "description: " + fields["title"]
                + " " + fields["bullets"] + " " + fields["desc"]).str.replace(_WS, " ", regex=True).str.strip()
    cols = [c for c in ORDERS[order] if c in fields]
    return fields[cols].agg(f" {sep} ".join, axis=1)


def budget_fields(fields, tokenizer, budget):
    """Field budgets in tokens (docs/2.md proposal 6), e.g. {"title": None, "bullets": 96, "desc": 96}:
    each field is cut to its budget before joining, instead of first-come truncation of the joined text."""
    out = fields.copy()
    for f, n in budget.items():
        if n is None or f not in out:
            continue
        enc = tokenizer(out[f].tolist(), add_special_tokens=False, truncation=True, max_length=n)["input_ids"]
        out[f] = tokenizer.batch_decode(enc, skip_special_tokens=True)
    return out


def mark_brands(queries, brands):
    """[www] typed entity markers, reduced to brands from the catalogue: 'sony tv' -> '[B] sony [/B] tv'.
    (www also marked product mentions with an NER tagger; that half is not built.)"""
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, brands), key=len, reverse=True)) + r")\b", re.IGNORECASE)
    return [pat.sub(lambda m: f"[B] {m.group(0)} [/B]", q) for q in queries]
