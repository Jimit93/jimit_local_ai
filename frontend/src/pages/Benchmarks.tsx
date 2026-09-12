import { useEffect, useState } from "react";
import { api } from "../services/api";
import { Card, Err, Metric, Spinner, fmtMs } from "../components/Bits";

function pct(x: number | null | undefined) {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(1)}%`;
}

export default function Benchmarks() {
  const [rep, setRep] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  useEffect(() => {
    api.benchmarkLatest().then(setRep).catch(() => setNote("No benchmark has been run yet."));
  }, []);

  async function run(includeBaseline: boolean) {
    setBusy(true);
    setError("");
    setNote(
      includeBaseline
        ? "Running both modes. The LLM-only baseline calls the model for every task, so this takes several minutes on CPU."
        : "Running the harness only.",
    );
    try {
      setRep(await api.benchmarkRun({ include_baseline: includeBaseline }));
      setNote("");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const h = rep?.totals?.harness;
  const b = rep?.totals?.baseline;
  const c = rep?.comparison || {};

  return (
    <>
      <div className="page-head">
        <h2>Benchmarks</h2>
        <p>Measured comparison: LLM-only baseline versus the routing harness, same model.</p>
      </div>

      <Card>
        <div className="btn-row">
          <button className="btn" disabled={busy} onClick={() => run(true)}>
            Run full benchmark (with baseline)
          </button>
          <button className="btn ghost" disabled={busy} onClick={() => run(false)}>
            Harness only (fast)
          </button>
          {busy && <Spinner label="benchmarking…" />}
        </div>
        {note && <p className="muted" style={{ marginTop: 10 }}>{note}</p>}
        {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
      </Card>

      {rep && (
        <>
          <Card title="Environment">
            <p className="mono muted">
              {rep.environment.model} ({rep.environment.model_size_gb} GB) ·{" "}
              {rep.environment.runtime} · n_ctx {rep.environment.n_ctx} ·{" "}
              {rep.environment.n_threads} threads · {rep.environment.cpu_logical} cores ·{" "}
              {rep.environment.ram_total_gb} GB RAM · GPU {String(rep.environment.gpu)} ·
              run {rep.run_id} · {rep.duration_s}s
            </p>
          </Card>

          {b && (
            <Card title="Harness vs baseline">
              <div className="metrics">
                <Metric k="LLM call reduction" v={pct(c.llm_call_reduction)} />
                <Metric k="Token reduction" v={pct(c.token_reduction)} />
                <Metric k="Latency reduction" v={pct(c.latency_reduction)} />
                <Metric k="CPU reduction" v={pct(c.cpu_reduction)} />
                <Metric k="Accuracy delta"
                  v={`${(c.accuracy_delta * 100).toFixed(1)} pts`} />
                <Metric k="No-LLM tasks" v={`${h.llm_free_tasks}/${h.tasks}`} />
              </div>
              <div className="scroll-x" style={{ marginTop: 16 }}>
                <table>
                  <thead>
                    <tr><th>Metric</th><th className="num">Baseline (LLM only)</th>
                      <th className="num">Harness</th></tr>
                  </thead>
                  <tbody>
                    <tr><td>Accuracy</td><td className="num">{pct(b.accuracy)}</td>
                      <td className="num">{pct(h.accuracy)}</td></tr>
                    <tr><td>LLM calls</td><td className="num">{b.llm_calls}</td>
                      <td className="num">{h.llm_calls}</td></tr>
                    <tr><td>Tokens</td><td className="num">{b.tokens}</td>
                      <td className="num">{h.tokens}</td></tr>
                    <tr><td>Total latency</td><td className="num">{fmtMs(b.total_latency_ms)}</td>
                      <td className="num">{fmtMs(h.total_latency_ms)}</td></tr>
                    <tr><td>Median latency</td><td className="num">{fmtMs(b.median_latency_ms)}</td>
                      <td className="num">{fmtMs(h.median_latency_ms)}</td></tr>
                    <tr><td>CPU time (all threads)</td>
                      <td className="num">{fmtMs(b.total_cpu_ms)}</td>
                      <td className="num">{fmtMs(h.total_cpu_ms)}</td></tr>
                    <tr><td>Peak RSS</td><td className="num">{b.peak_rss_mb} MB</td>
                      <td className="num">{h.peak_rss_mb} MB</td></tr>
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          <Card title="By suite">
            <div className="scroll-x">
              <table>
                <thead>
                  <tr><th>Suite</th><th className="num">Harness acc.</th>
                    <th className="num">Baseline acc.</th><th className="num">Harness LLM</th>
                    <th className="num">Baseline LLM</th><th className="num">Harness time</th>
                    <th className="num">Baseline time</th></tr>
                </thead>
                <tbody>
                  {Object.entries(rep.by_suite || {}).map(([s, v]: [string, any]) => (
                    <tr key={s}>
                      <td>{s}</td>
                      <td className="num">{pct(v.harness.accuracy)}</td>
                      <td className="num">{v.baseline ? pct(v.baseline.accuracy) : "—"}</td>
                      <td className="num">{v.harness.llm_calls}</td>
                      <td className="num">{v.baseline ? v.baseline.llm_calls : "—"}</td>
                      <td className="num">{fmtMs(v.harness.total_latency_ms)}</td>
                      <td className="num">
                        {v.baseline ? fmtMs(v.baseline.total_latency_ms) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Per task">
            <div className="scroll-x">
              <table>
                <thead>
                  <tr><th>Task</th><th>Mode</th><th>Correct</th><th className="num">Score</th>
                    <th>Processor</th><th className="num">LLM</th><th className="num">Tokens</th>
                    <th className="num">Latency</th></tr>
                </thead>
                <tbody>
                  {(rep.results || []).map((r: any, i: number) => (
                    <tr key={i}>
                      <td className="mono">{r.task_id}</td>
                      <td>{r.mode}</td>
                      <td>{r.correct ? "✓" : "✗"}</td>
                      <td className="num">{r.score.toFixed(2)}</td>
                      <td className="mono">{(r.processor || "—").slice(0, 28)}</td>
                      <td className="num">{r.llm_calls}</td>
                      <td className="num">{r.tokens}</td>
                      <td className="num">{fmtMs(r.latency_ms)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </>
  );
}
