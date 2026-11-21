"""Response models (backend.md §9.4). Mirrored by web/src/types/api.ts; keep the two in sync."""
from typing import Literal

from pydantic import BaseModel

Label = Literal["E", "S", "C", "I"]


class Product(BaseModel):
    product_id: str
    title: str | None
    brand: str | None
    color: str | None
    bullets: str | None
    image_url: str | None


class Probs(BaseModel):
    E: float
    S: float
    C: float
    I: float  # noqa: E741  (ESCI label name)


class ProductHit(Product):
    score: float
    probs: Probs | None = None  # null until a reranker is plugged in (docs/demo-handoff.md)
    predicted: Label | None = None
    human_label: Label | None = None
    rank_text_only: int | None = None


class Timings(BaseModel):
    retrieve: float
    rerank: float
    total: float


class SearchResponse(BaseModel):
    query: str
    mode: Literal["multimodal", "text"]
    retriever: str
    reranker: str | None
    items: list[ProductHit]
    matched_query_id: int | None
    ground_truth: bool
    timings_ms: Timings


class HitList(BaseModel):
    items: list[ProductHit]


class ModelInfo(BaseModel):
    name: str
    role: str
    dev_ndcg: float | None
    ci: list[float] | None
    test_ndcg: float | None
    ms_per_query: float | None = None


class CompareItem(Product):
    human_label: Label
    zr_rank: int
    ours_rank: int
    zr_score: float
    ours_score: float


class CompareModels(BaseModel):
    zr: str
    ours: str


class CompareNdcg(BaseModel):
    zr: float
    ours: float


class CompareResponse(BaseModel):
    query: str
    query_id: int
    models: CompareModels
    items: list[CompareItem]  # ordered by ours_rank
    ndcg: CompareNdcg


class Example(BaseModel):
    query: str
    reason: str


class Health(BaseModel):
    status: str
    db: str
    products: int
    models: list[str]
    reranker: str | None
    gpu: bool
