import { useEffect, useState } from "react";
import { api } from "../services/api";
import { Card, Err, LayerPill, Metric, Spinner, fmtMs } from "../components/Bits";

export default function Tasks() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");

  async function load() {
    try {
      setData(await api.metrics());
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  if (error) return <Err>{error}</Err>;
  if (!data) return <Spinner />;

  const s = data.summary;

  return (
    <>
      <div className="page-head">
        <h2>Tasks</h2>
        <p>Everything the harness has executed on this machine, with its real cost.</p>
      </div>

      <Card title="Lifetime totals">
        <div className="metrics">
          <Metric k="Tasks" v={s.tasks} />
          <Metric k="LLM calls" v={s.llm_calls} />
          <Metric k="Tokens" v={s.tokens} />
          <Metric k="Answered without LLM" v={`${Math.round((s.llm_free_rate || 0) * 100)}%`} />
          <Metric k="Success rate" v={`${Math.round((s.success_rate || 0) * 100)}%`} />
          <Metric k="Median latency" v={fmtMs(s.median_latency_ms)} />
          <Metric k="Mean latency" v={fmtMs(s.avg_latency_ms)} />
          <Metric k="Escalations" v={s.escalations} />
        </div>
      </Card>

      <div className="grid-2">
        <Card title="By layer">
          <table>
            <thead><tr><th>Layer</th><th className="num">Tasks</th></tr></thead>
            <tbody>
              {Object.entries(s.by_layer || {}).map(([k, v]) => (
                <tr key={k}><td><LayerPill layer={k} /></td><td className="num">{String(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </Card>
        <Card title="By processor">
          <table>
            <thead><tr><th>Processor</th><th className="num">Tasks</th></tr></thead>
            <tbody>
              {Object.entries(s.by_processor || {}).map(([k, v]) => (
                <tr key={k}><td className="mono">{k}</td><td className="num">{String(v)}</td></tr>
              ))}
            </tbody>
          </table>
        </Card>
      </div>

      <Card title="Recent runs">
        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>Time</th><th>Type</th><th>Processor</th><th>Layer</th>
                <th className="num">Conf.</th><th className="num">Latency</th>
                <th className="num">LLM</th><th className="num">Tokens</th><th>OK</th></tr>
            </thead>
            <tbody>
              {(data.recent || []).map((r: any) => (
                <tr key={r.id}>
                  <td className="mono">{new Date(r.ts * 1000).toLocaleTimeString()}</td>
                  <td>{r.task_type}</td>
                  <td className="mono">{(r.processor || "").slice(0, 30)}</td>
                  <td>{r.layer}</td>
                  <td className="num">{(r.confidence ?? 0).toFixed(2)}</td>
                  <td className="num">{fmtMs(r.latency_ms)}</td>
                  <td className="num">{r.llm_calls}</td>
                  <td className="num">{r.tokens}</td>
                  <td>{r.success ? "✓" : "✗"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
