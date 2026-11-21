import { Link, useParams, useSearchParams } from "react-router-dom";
import { useApi } from "@/api/client";
import type { Example, HitList, Mode, Product, SearchResponse } from "@/types/api";
import { ErrorBanner, Header, Loading, ModelFooter, ResultsGrid, Sidebar } from "@/components";

const HERO_QUERY = "women sunglasses"; // only picks the poster photos

function Hero() {
  const { data } = useApi<SearchResponse>(`/api/search?q=${encodeURIComponent(HERO_QUERY)}&k=12`);
  const pics = (data?.items ?? []).filter((h) => h.image_url).slice(0, 3);
  return (
    <section className="hero">
      <div className="hero-text">
        <p className="eyebrow">Multimodal Search Engine</p>
        <h1>
          SEARCH BY
          <br />
          TEXT &amp; IMAGE
        </h1>
        <p className="hero-sub">
          A product search engine trained on Amazon's ESCI dataset, with product photos from SQID. Results carry the
          human E/S/C/I labels, so you can see how good each ranking really is.
        </p>
        <a className="btn" href="#queries">
          Try a real query
        </a>
      </div>
      <div className="hero-pics">
        {pics.map((h, i) => (
          <Link key={h.product_id} to={`/product/${h.product_id}`} className={`hero-pic p${i}`}>
            <img src={h.image_url ?? ""} alt={h.title ?? ""} referrerPolicy="no-referrer" />
          </Link>
        ))}
      </div>
    </section>
  );
}

function Featured() {
  const ex = useApi<Example[]>("/api/examples");
  const q = ex.data?.[0]?.query ?? null;
  const res = useApi<SearchResponse>(q ? `/api/search?q=${encodeURIComponent(q)}&k=8` : null);
  if (!q) return null;
  return (
    <section id="featured">
      <div className="section-head">
        <h2 className="section-title">Featured query: “{q}”</h2>
        <Link to={`/search?q=${encodeURIComponent(q)}`}>See all results →</Link>
      </div>
      {res.data && <ResultsGrid items={res.data.items} scoreLabel="bm25" />}
    </section>
  );
}

export function HomePage() {
  return (
    <>
      <Header />
      <main className="container">
        <Hero />
        <div className="layout">
          <Sidebar />
          <div className="main-col">
            <Featured />
          </div>
        </div>
      </main>
      <ModelFooter />
    </>
  );
}

export function SearchPage() {
  const [params, setParams] = useSearchParams();
  const q = params.get("q") ?? "";
  const mode: Mode = params.get("mode") === "text" ? "text" : "multimodal";
  const url = q ? `/api/search?q=${encodeURIComponent(q)}&mode=${mode}` : null;
  const { data, error, loading } = useApi<SearchResponse>(url);
  const judged = data?.items.filter((h) => h.human_label).length ?? 0;
  const noReranker = data !== null && data.reranker === null;

  return (
    <>
      <Header query={q} />
      <main className="container layout">
        <Sidebar />
        <div className="main-col">
          <div className="section-head">
            <h2 className="section-title">Results for “{q}”</h2>
          </div>
          <div className="toolbar">
            <div className="toggle" role="group" aria-label="Ranking mode">
              {(["multimodal", "text"] as const).map((m) => (
                <button
                  key={m}
                  className={m === mode ? "on" : ""}
                  disabled={noReranker}
                  title={noReranker ? "Needs the trained rerankers (Phases 4–5)" : undefined}
                  onClick={() => setParams({ q, mode: m })}
                >
                  {m === "multimodal" ? "Multimodal" : "Text only"}
                </button>
              ))}
            </div>
            {data && (
              <span className="timings">
                {data.retriever}
                {data.reranker ? ` → ${data.reranker} reranker` : " · no reranker yet"} · retrieve{" "}
                {data.timings_ms.retrieve} ms · rerank {data.timings_ms.rerank} ms · total {data.timings_ms.total} ms
              </span>
            )}
          </div>
          {data?.ground_truth && (
            <div className="banner info">
              Real ESCI query: human labels shown on the {judged} of {data.items.length} results that annotators
              judged for it.
            </div>
          )}
          {error && <ErrorBanner error={error} />}
          {loading && !data && <Loading />}
          {data && data.items.length === 0 && (
            <p className="muted pad">No products match “{q}”. Try a real ESCI query from the list.</p>
          )}
          {data && <ResultsGrid items={data.items} scoreLabel={data.reranker ? "score" : "bm25"} />}
        </div>
      </main>
      <ModelFooter />
    </>
  );
}

function Row({ title, url, scoreLabel }: { title: string; url: string; scoreLabel: string }) {
  const { data, error } = useApi<HitList>(url);
  return (
    <section className="row">
      <h2 className="section-title">{title}</h2>
      {error && (
        <p className="muted">
          {error.code === "MODEL_NOT_READY" ? `Coming after training: ${error.message}.` : error.message}
        </p>
      )}
      {data && data.items.length === 0 && <p className="muted">Nothing to show (this product has no photo).</p>}
      {data && data.items.length > 0 && <ResultsGrid items={data.items} scoreLabel={scoreLabel} />}
    </section>
  );
}

export function ProductPage() {
  const { id = "" } = useParams();
  const base = `/api/products/${encodeURIComponent(id)}`;
  const { data, error } = useApi<Product>(base);
  return (
    <>
      <Header />
      <main className="container">
        {error && <ErrorBanner error={error} />}
        {data && (
          <article className="detail">
            <div className="detail-img">
              {data.image_url ? (
                <img src={data.image_url} alt={data.title ?? ""} referrerPolicy="no-referrer" />
              ) : (
                <span className="noimg">no image</span>
              )}
            </div>
            <div className="detail-text">
              {data.brand && <span className="brand">{data.brand}</span>}
              <h1>{data.title}</h1>
              {data.color && <p>Color: {data.color}</p>}
              {data.bullets && <p className="bullets">{data.bullets}…</p>}
              <a className="btn" href={`https://www.amazon.com/dp/${data.product_id}`} target="_blank" rel="noreferrer">
                View on Amazon
              </a>
              <p className="muted small">ASIN {data.product_id}</p>
            </div>
          </article>
        )}
        {data && (
          <>
            <Row key={`ll-${id}`} title="Looks similar (CLIP image similarity)" url={`${base}/looks-like`} scoreLabel="cos" />
            <Row key={`alt-${id}`} title="Alternatives" url={`${base}/alternatives`} scoreLabel="P(S)" />
            <Row key={`gw-${id}`} title="Goes with" url={`${base}/goes-with`} scoreLabel="P(C)" />
          </>
        )}
      </main>
      <ModelFooter />
    </>
  );
}

export function NotFoundPage() {
  return (
    <>
      <Header />
      <main className="container pad">
        <h1>Page not found</h1>
        <Link to="/">Back to search</Link>
      </main>
    </>
  );
}
