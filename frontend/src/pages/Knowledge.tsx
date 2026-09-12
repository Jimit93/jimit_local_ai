import { useEffect, useState } from "react";
import { api, type Hit } from "../services/api";
import { Card, Err, Metric, Spinner, fmtMs } from "../components/Bits";

export default function Knowledge() {
  const [stats, setStats] = useState<any>(null);
  const [q, setQ] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [hits, setHits] = useState<Hit[]>([]);
  const [meta, setMeta] = useState<{ latency_ms: number; count: number } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api.knowledge().then(setStats).catch((e) => setError(String(e)));
  }, []);

  async function search() {
    if (!q.trim()) return;
    setBusy(true);
    setError("");
    try {
      const r = await api.search({ q, mode, top_k: 10 });
      setHits(r.hits);
      setMeta({ latency_ms: r.latency_ms, count: r.count });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const byType = stats?.by_type || {};
  const topics = stats?.topics || {};

  return (
    <>
      <div className="page-head">
        <h2>Knowledge</h2>
        <p>Your LinkedIn export, parsed locally into a searchable knowledge base.</p>
      </div>

      {error && <Err>{error}</Err>}

      <Card title="Corpus">
        {!stats ? <Spinner /> : (
          <>
            <div className="metrics">
              <Metric k="Posts" v={byType.post ?? 0} />
              <Metric k="Comments" v={byType.comment ?? 0} />
              <Metric k="Articles" v={byType.article ?? 0} />
              <Metric k="Messages (own)" v={byType.message ?? 0} />
              <Metric k="Reactions" v={byType.reaction ?? 0} />
              <Metric k="Embeddings" v={stats.retrieval?.embeddings ?? 0} />
              <Metric k="Style features" v={stats.documents?.style_features ?? 0} />
              <Metric k="Indexed" v={stats.retrieval?.documents_indexed ?? 0} />
              <Metric k="Semantic chunks" v={stats.retrieval?.chunks ?? 0} />
              <Metric k="Split documents" v={stats.retrieval?.multi_chunk_docs ?? 0} />
            </div>
            <p className="muted mono" style={{ marginTop: 12 }}>
              {stats.retrieval?.model} · {stats.retrieval?.dim}d · FAISS{" "}
              {String(stats.retrieval?.faiss)} · {stats.retrieval?.chunking} chunking (
              {stats.retrieval?.chunks_per_doc} chunks/doc) · index build{" "}
              {fmtMs(stats.retrieval?.build_ms)}
            </p>
            {stats.style_profile?.summary && (
              <p style={{ marginTop: 8, fontSize: 13 }}>{stats.style_profile.summary}</p>
            )}
          </>
        )}
      </Card>

      <Card title="Topics">
        {Object.keys(topics).length === 0 ? <p className="muted">No topics yet.</p> : (
          <div className="metrics">
            {Object.entries(topics).map(([t, n]) => (
              <Metric key={t} k={t.replace(/_/g, " ")} v={String(n)} />
            ))}
          </div>
        )}
      </Card>

      <Card title="Search">
        <div className="btn-row">
          <input value={q} onChange={(e) => setQ(e.target.value)} style={{ flex: 3 }}
            placeholder="Search your own writing…"
            onKeyDown={(e) => e.key === "Enter" && search()} />
          <select value={mode} onChange={(e) => setMode(e.target.value)} style={{ maxWidth: 190 }}>
            <option value="hybrid">Hybrid (BM25 + vectors)</option>
            <option value="keyword">Exact keyword</option>
            <option value="semantic">Semantic only</option>
          </select>
          <button className="btn" onClick={search} disabled={busy}>Search</button>
        </div>
        {busy && <div style={{ marginTop: 10 }}><Spinner /></div>}
        {meta && (
          <p className="muted mono" style={{ marginTop: 10 }}>
            {meta.count} hit(s) in {fmtMs(meta.latency_ms)} · no LLM call
          </p>
        )}
        <div style={{ marginTop: 10 }}>
          {hits.map((h) => (
            <div key={h.doc_id} className="bubble" style={{ marginBottom: 10 }}>
              <div className="head">
                <span className="pill nlp">{h.doc_type}</span>
                {h.topic && <span className="pill ml">{h.topic}</span>}
                <span className="mono">{h.created_at || "undated"}</span>
                <span className="mono">
                  score {h.score} (lex {h.lexical} / sem {h.semantic})
                </span>
                {h.n_chunks > 1 && (
                  <span className="pill retrieval">
                    chunk {h.chunk_index + 1}/{h.n_chunks}
                  </span>
                )}
              </div>
              <div className="answer" style={{ fontSize: 13 }}>{h.text}</div>
            </div>
          ))}
        </div>
      </Card>
    </>
  );
}
