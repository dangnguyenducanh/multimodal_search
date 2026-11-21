import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useApi } from "@/api/client";
import type { ApiError } from "@/api/client";
import type { Example, Label, ModelInfo, ProductHit } from "@/types/api";

export const LABELS: Record<Label, string> = { E: "Exact", S: "Substitute", C: "Complement", I: "Irrelevant" };
const REASON: Record<Example["reason"], string> = { gained: "images helped", lost: "images hurt", random: "random" };

export function SearchBar({ initial = "", big = false }: { initial?: string; big?: boolean }) {
  const [q, setQ] = useState(initial);
  const nav = useNavigate();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (q.trim()) nav(`/search?q=${encodeURIComponent(q.trim())}`);
  };
  return (
    <form className={big ? "searchbar big" : "searchbar"} onSubmit={submit} role="search">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="Enter a product or a real ESCI query…"
        maxLength={200}
        aria-label="Search products"
      />
      <button type="submit" aria-label="Search">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" strokeWidth="2" />
          <path d="M20 20l-4-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      </button>
    </form>
  );
}

export function Header({ query }: { query?: string }) {
  return (
    <header className="header">
      <div className="notice">
        <div className="container notice-row">
          <span>Amazon ESCI · KDD Cup 2022</span>
          <span>
            <b>164,900</b> products · <b>8,956</b> real queries with human labels
          </span>
          <span>EN · US</span>
        </div>
      </div>
      <div className="container header-main">
        <Link to="/" className="logo">
          <span className="logo-name">
            multimodal <span>search engine</span>
          </span>
          <small className="logo-sub">Duc Anh's personal project</small>
        </Link>
        <SearchBar key={query ?? ""} initial={query ?? ""} />
      </div>
      <nav className="nav" aria-label="Main">
        <a href="/">Home</a>
        <a href="/#queries">Real queries</a>
        <a href="/#featured">Featured</a>
        <a href="#models">Models</a>
      </nav>
    </header>
  );
}

export function LabelBadge({ label, kind }: { label: Label; kind: "human" | "model" }) {
  return (
    <span className={`badge badge-${label}`} title={kind === "human" ? "ESCI human label" : "model prediction"}>
      {kind === "human" ? "Human: " : "Model: "}
      {LABELS[label]}
    </span>
  );
}

export function ProductCard({ hit, scoreLabel }: { hit: ProductHit; scoreLabel: string }) {
  return (
    <Link to={`/product/${hit.product_id}`} className="card">
      <div className="card-img">
        {hit.image_url ? (
          <img src={hit.image_url} alt={hit.title ?? ""} loading="lazy" referrerPolicy="no-referrer" />
        ) : (
          <span className="noimg">no image</span>
        )}
        {hit.human_label && <span className={`ribbon ribbon-${hit.human_label}`}>{LABELS[hit.human_label]}</span>}
      </div>
      <div className="card-body">
        <span className="brand">{hit.brand || "—"}</span>
        <h3 className="title">{hit.title}</h3>
        <div className="badges">
          {hit.predicted && <LabelBadge label={hit.predicted} kind="model" />}
          {hit.human_label && <LabelBadge label={hit.human_label} kind="human" />}
          {hit.rank_text_only !== null && <span className="moved">text-only rank {hit.rank_text_only}</span>}
        </div>
        <span className="score">
          <b>{hit.score.toFixed(2)}</b> {scoreLabel}
        </span>
      </div>
    </Link>
  );
}

export function ResultsGrid({ items, scoreLabel }: { items: ProductHit[]; scoreLabel: string }) {
  return (
    <div className="grid">
      {items.map((h) => (
        <ProductCard key={h.product_id} hit={h} scoreLabel={scoreLabel} />
      ))}
    </div>
  );
}

/** Left column, after Anon's "Category" + "Best sellers" boxes. */
export function Sidebar() {
  const { data } = useApi<Example[]>("/api/examples");
  return (
    <aside className="sidebar" id="queries">
      <div className="box">
        <h3 className="box-title">Real ESCI queries</h3>
        <ul className="side-list">
          {(data ?? []).map((e) => (
            <li key={e.query}>
              <Link to={`/search?q=${encodeURIComponent(e.query)}`}>
                <span>{e.query}</span>
                <small className={`tag tag-${e.reason}`}>{REASON[e.reason]}</small>
              </Link>
            </li>
          ))}
        </ul>
      </div>
      <div className="box">
        <h3 className="box-title">ESCI labels</h3>
        <ul className="legend">
          <li><span className="badge badge-E">Exact</span> what was asked for</li>
          <li><span className="badge badge-S">Substitute</span> could replace it</li>
          <li><span className="badge badge-C">Complement</span> goes with it</li>
          <li><span className="badge badge-I">Irrelevant</span> wrong product</li>
        </ul>
        <p className="muted small">nDCG gains: E 1 · S 0.1 · C 0.01 · I 0</p>
      </div>
    </aside>
  );
}

export function ErrorBanner({ error }: { error: ApiError }) {
  return (
    <div className="banner error" role="alert">
      {error.message}
    </div>
  );
}

export function Loading() {
  return <div className="muted pad">Loading…</div>;
}

export function ModelFooter() {
  const { data } = useApi<ModelInfo[]>("/api/models");
  return (
    <footer className="footer" id="models">
      <div className="container footer-grid">
        <div>
          <h4>Models</h4>
          {data && data.length > 0 ? (
            <ul>
              {data.map((m) => (
                <li key={m.name}>
                  {m.name} ({m.role}): dev nDCG {m.dev_ndcg?.toFixed(4) ?? "–"}
                  {m.ci && ` [${m.ci[0].toFixed(4)}, ${m.ci[1].toFixed(4)}]`}
                  {m.test_ndcg !== null && ` · test ${m.test_ndcg.toFixed(4)}`}
                </li>
              ))}
            </ul>
          ) : (
            <p>
              No trained models yet: results are BM25 retrieval only. Dev nDCG ± CI of each trained reranker appears
              here once Phases 4–6 finish.
            </p>
          )}
        </div>
        <div>
          <h4>Data</h4>
          <p>
            Amazon ESCI Shopping Queries (US test products) with SQID product photos. E/S/C/I are ESCI's human
            judgments.
          </p>
        </div>
        <div>
          <h4>Method</h4>
          <p>Built on the KDD Cup 2022 1st-place (www) and 5th-place (ZhichunRoad) solutions.</p>
        </div>
      </div>
    </footer>
  );
}
