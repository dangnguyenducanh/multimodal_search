// Mirrors app/schemas.py (backend.md §9.4). Change both together.
export type Label = "E" | "S" | "C" | "I";
export type Mode = "multimodal" | "text";

export interface Product {
  product_id: string;
  title: string | null;
  brand: string | null;
  color: string | null;
  bullets: string | null;
  image_url: string | null;
}

export interface ProductHit extends Product {
  score: number;
  probs: Record<Label, number> | null;
  predicted: Label | null;
  human_label: Label | null;
  rank_text_only: number | null;
}

export interface SearchResponse {
  query: string;
  mode: Mode;
  retriever: string;
  reranker: string | null;
  items: ProductHit[];
  matched_query_id: number | null;
  ground_truth: boolean;
  timings_ms: { retrieve: number; rerank: number; total: number };
}

export interface HitList {
  items: ProductHit[];
}

export interface ModelInfo {
  name: string;
  role: string;
  dev_ndcg: number | null;
  ci: [number, number] | null;
  test_ndcg: number | null;
  ms_per_query?: number | null;
}

export interface CompareItem extends Product {
  human_label: Label;
  zr_rank: number;
  ours_rank: number;
  zr_score: number;
  ours_score: number;
}

/** GET /api/compare?q= ; 404 NO_BASELINE_FOR_QUERY when the query has no saved ZR + ours scores. */
export interface CompareResponse {
  query: string;
  query_id: number;
  models: { zr: string; ours: string };
  items: CompareItem[]; // ordered by ours_rank
  ndcg: { zr: number; ours: number };
}

export interface Example {
  query: string;
  reason: "gained" | "lost" | "random";
}

export interface ApiErrorBody {
  error: { code: string; message: string; details: { path: string; issue: string }[] };
}
