import { useState } from "react";
import { api, type TaskResult } from "../services/api";
import { Bar, Card, Err, LayerPill, Metric, Spinner, StatusPill, fmtMs } from "../components/Bits";

interface Turn { task: string; res?: TaskResult; error?: string }

const SAMPLES = [
  "What is 25 * 40?",
  "How many posts do I have?",
  "Find all LinkedIn posts mentioning RAG",
  "Find posts semantically related to local AI",
  "What are the top 5 hashtags I use?",
];

export default function Chat() {
  const [input, setInput] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState("harness");

  async function run(text?: string) {
    const task = (text ?? input).trim();
    if (!task || busy) return;
    setBusy(true);
    setInput("");
    setTurns((t) => [...t, { task }]);
    try {
      const res = await api.task(task, mode);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, res } : x)));
    } catch (e) {
      const error = String(e);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error } : x)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <div className="page-head">
        <h2>Chat</h2>
        <p>Every task is routed to the cheapest processor that can answer it reliably.</p>
      </div>

      <Card>
        <div className="field">
          <textarea
            rows={3}
            value={input}
            placeholder="Ask anything — arithmetic, corpus questions, search, analysis…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
            }}
          />
        </div>
        <div className="btn-row">
          <button className="btn" disabled={busy || !input.trim()} onClick={() => run()}>
            {busy ? "Running…" : "Run"}
          </button>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            style={{ width: 210 }}
            title="Baseline sends the raw task straight to the model, with no routing."
          >
            <option value="harness">Harness (routed)</option>
            <option value="baseline">Baseline (LLM only)</option>
          </select>
          <span className="muted mono">Ctrl+Enter to run</span>
        </div>
        <div className="btn-row" style={{ marginTop: 12 }}>
          {SAMPLES.map((s) => (
            <button key={s} className="btn ghost" style={{ fontSize: 12, padding: "6px 10px" }}
              disabled={busy} onClick={() => run(s)}>
              {s}
            </button>
          ))}
        </div>
      </Card>

      {[...turns].reverse().map((t, i) => (
        <div key={turns.length - i} className="bubble">
          <div className="head"><strong style={{ color: "var(--text)" }}>You</strong></div>
          <div className="answer">{t.task}</div>

          <div style={{ marginTop: 14, borderTop: "1px solid var(--line)", paddingTop: 14 }}>
            {!t.res && !t.error && <Spinner label="routing…" />}
            {t.error && <Err>{t.error}</Err>}
            {t.res && <Answer res={t.res} />}
          </div>
        </div>
      ))}
    </>
  );
}

function Answer({ res }: { res: TaskResult }) {
  const hits = (res.result as any)?.hits as any[] | undefined;
  return (
    <>
      <div className="head">
        <LayerPill layer={res.layer} />
        <StatusPill status={res.status} />
        <span className="mono">{res.processor}</span>
      </div>

      <div className="answer">{res.answer || <span className="muted">no answer</span>}</div>

      <div className="metrics" style={{ marginTop: 14 }}>
        <Metric k="Processor" v={<span style={{ fontSize: 12 }}>{res.task_type}</span>} small />
        <Metric k="Confidence" v={res.confidence.toFixed(2)} />
        <Metric k="Latency" v={fmtMs(res.latency_ms)} />
        <Metric k="LLM call" v={res.llm_used ? `YES (${res.llm_calls})` : "NO"} />
        <Metric k="Tokens" v={res.tokens} />
        <Metric k="Escalations" v={res.escalations} />
      </div>
      <div style={{ marginTop: 8 }}><Bar value={res.confidence} /></div>

      {hits && hits.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary>{hits.length} source document(s)</summary>
          {hits.map((h) => (
            <div key={h.doc_id} className="mono" style={{ marginBottom: 8 }}>
              <span className="muted">
                [{h.created_at || "undated"} · {h.doc_type} · {h.score}]
              </span>{" "}
              {h.text}
            </div>
          ))}
        </details>
      )}

      <details style={{ marginTop: 12 }}>
        <summary>Execution trace ({res.steps.length} step{res.steps.length === 1 ? "" : "s"})</summary>
        <div className="scroll-x">
          <table>
            <thead>
              <tr><th>Processor</th><th>Layer</th><th className="num">Confidence</th>
                <th className="num">Latency</th><th>Accepted</th><th>Error</th></tr>
            </thead>
            <tbody>
              {res.steps.map((s, i) => (
                <tr key={i}>
                  <td className="mono">{s.processor}</td>
                  <td>{s.layer || "—"}</td>
                  <td className="num">{s.confidence.toFixed(2)}</td>
                  <td className="num">{fmtMs(s.latency_ms)}</td>
                  <td>{s.accepted ? "yes" : "no"}</td>
                  <td className="muted">{s.error || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="muted mono" style={{ marginTop: 8 }}>
          classified by {res.classifier.processor} ({res.classifier.confidence.toFixed(2)}) ·
          RSS {res.resources.rss_mb} MB
        </div>
      </details>
    </>
  );
}
