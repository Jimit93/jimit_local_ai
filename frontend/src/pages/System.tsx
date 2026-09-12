import { useEffect, useState } from "react";
import { api } from "../services/api";
import { Card, Err, Metric, Spinner, fmtMs } from "../components/Bits";

export default function System() {
  const [sys, setSys] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [ingest, setIngest] = useState<any>(null);

  async function load() {
    try {
      const [s, h] = await Promise.all([api.system(), api.health()]);
      setSys(s);
      setHealth(h);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  async function reingest() {
    setBusy(true);
    try {
      setIngest(await api.ingest());
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  if (error) return <Err>{error}</Err>;
  if (!sys) return <Spinner />;

  const sel = sys.selected_model;

  return (
    <>
      <div className="page-head">
        <h2>System</h2>
        <p>What this machine is running, measured live.</p>
      </div>

      <Card title="Runtime">
        <div className="metrics">
          <Metric k="Model" v={<span style={{ fontSize: 12 }}>{sel?.name || "—"}</span>} small />
          <Metric k="Loaded" v={health?.llm_loaded ? "YES" : "NO"} />
          <Metric k="Warm-up" v={health?.warmup?.state || "—"} small />
          <Metric k="Quant" v={sel?.quant || "—"} small />
          <Metric k="Size" v={sel ? `${(sel.total_size_bytes / 1e9).toFixed(2)} GB` : "—"} />
          <Metric k="Context" v={sys.llm_settings.n_ctx} />
          <Metric k="Threads" v={sys.llm_settings.n_threads} />
        </div>
        {health?.warmup?.llm_load_ms && (
          <p className="muted mono" style={{ marginTop: 10 }}>
            model load {fmtMs(health.warmup.llm_load_ms)} · index + embedder{" "}
            {fmtMs(health.warmup.index_build_ms)}
          </p>
        )}
      </Card>

      <Card title="Machine">
        <div className="metrics">
          <Metric k="CPU cores" v={`${sys.cpu_physical} / ${sys.cpu_logical}`} />
          <Metric k="RAM total" v={`${sys.ram_total_gb} GB`} />
          <Metric k="RAM free" v={`${sys.ram_available_gb} GB`} />
          <Metric k="GPU" v={sys.gpu ? "yes" : "none"} />
          <Metric k="Process RSS" v={`${health?.system?.process_rss_mb ?? "—"} MB`} />
          <Metric k="CPU now" v={`${health?.system?.cpu_percent ?? "—"}%`} />
          <Metric k="Python" v={sys.python} small />
        </div>
      </Card>

      <Card title="Models discovered">
        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>File</th><th>Family</th><th>Format</th><th>Quant</th>
                <th className="num">Size</th><th>Selected</th></tr>
            </thead>
            <tbody>
              {(sys.models_found || []).map((m: any) => (
                <tr key={m.path}>
                  <td className="mono">{m.name}</td>
                  <td>{m.family}</td>
                  <td>{m.fmt}</td>
                  <td>{m.quant || "—"}</td>
                  <td className="num">{(m.size_bytes / 1e9).toFixed(2)} GB</td>
                  <td>{sel && m.path === sel.path ? "✓" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Database">
        <div className="metrics">
          {Object.entries(sys.db || {}).map(([k, v]) => (
            <Metric key={k} k={k.replace(/_/g, " ")} v={String(v)} />
          ))}
        </div>
        <div className="btn-row" style={{ marginTop: 14 }}>
          <button className="btn ghost" onClick={reingest} disabled={busy}>
            {busy ? "Re-ingesting…" : "Re-ingest LinkedIn export"}
          </button>
          {busy && <Spinner label="parsing, embedding, retraining…" />}
        </div>
        {ingest && (
          <p className="muted mono" style={{ marginTop: 10 }}>
            inserted {ingest.ingest.inserted} · duplicates {ingest.ingest.duplicates} ·
            style docs {ingest.style?.n_documents} · index {ingest.index?.documents}
          </p>
        )}
      </Card>

      <Card title="Session LLM usage">
        <div className="metrics">
          <Metric k="Calls" v={sys.usage?.calls ?? 0} />
          <Metric k="Prompt tokens" v={sys.usage?.prompt_tokens ?? 0} />
          <Metric k="Completion tokens" v={sys.usage?.completion_tokens ?? 0} />
          <Metric k="LLM time" v={fmtMs(sys.usage?.latency_ms)} />
        </div>
      </Card>
    </>
  );
}
