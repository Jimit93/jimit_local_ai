import { useState } from "react";
import { api, type WriterResult } from "../services/api";
import { Bar, Card, Err, Metric, Spinner, StatusPill, fmtMs } from "../components/Bits";

export default function Writer() {
  const [topic, setTopic] = useState("");
  const [audience, setAudience] = useState("");
  const [pillar, setPillar] = useState("");
  const [length, setLength] = useState("medium");
  const [cta, setCta] = useState("");
  const [facts, setFacts] = useState("");
  const [instruction, setInstruction] = useState("");
  const [res, setRes] = useState<WriterResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState("");
  const [fbResult, setFbResult] = useState<any>(null);
  const [fbBusy, setFbBusy] = useState(false);

  async function sendFeedback(text: string) {
    if (!text.trim()) return;
    setFbBusy(true);
    setFbResult(null);
    try {
      setFbResult(await api.sendFeedback({ text, post: res?.text || "" }));
    } catch (e) {
      setError(String(e));
    } finally {
      setFbBusy(false);
    }
  }

  async function call(fn: () => Promise<WriterResult>) {
    setBusy(true);
    setError("");
    try {
      setRes(await fn());
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const generate = () =>
    call(() => api.generate({ topic, audience, pillar, length, cta, facts }));
  const improve = () =>
    call(() =>
      api.improve({ previous: res?.text || "", instruction: instruction || "tighten it", topic, length }),
    );

  function copy() {
    navigator.clipboard.writeText(res?.text || "");
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  }

  const v = res?.validation;

  return (
    <>
      <div className="page-head">
        <h2>LinkedIn Writer</h2>
        <p>
          Style profile and examples are retrieved locally; the model is called once, then the
          output is checked against your measured style.
        </p>
      </div>

      <div className="grid-2">
        <Card title="Brief">
          <div className="field">
            <label>Topic</label>
            <textarea rows={3} value={topic} onChange={(e) => setTopic(e.target.value)}
              placeholder="What the post is about" />
          </div>
          <div className="row">
            <div className="field">
              <label>Audience</label>
              <input value={audience} onChange={(e) => setAudience(e.target.value)}
                placeholder="AI engineers, founders…" />
            </div>
            <div className="field">
              <label>Content pillar</label>
              <input value={pillar} onChange={(e) => setPillar(e.target.value)}
                placeholder="engineering depth, proof…" />
            </div>
          </div>
          <div className="row">
            <div className="field">
              <label>Length</label>
              <select value={length} onChange={(e) => setLength(e.target.value)}>
                <option value="short">Short</option>
                <option value="medium">Medium</option>
                <option value="long">Long</option>
              </select>
            </div>
            <div className="field">
              <label>CTA</label>
              <select value={cta} onChange={(e) => setCta(e.target.value)}>
                <option value="">Auto (your usual)</option>
                <option value="comment">Comment</option>
                <option value="dm">DM</option>
                <option value="link">Link</option>
                <option value="follow">Follow</option>
                <option value="hiring">Hiring / opportunities</option>
              </select>
            </div>
          </div>
          <div className="field">
            <label>Facts the model may use (nothing else is allowed)</label>
            <textarea rows={4} value={facts} onChange={(e) => setFacts(e.target.value)}
              placeholder="Real numbers and details for this post. Old posts are used for style only, never as current fact." />
          </div>
          <div className="btn-row">
            <button className="btn" disabled={busy || topic.trim().length < 3} onClick={generate}>
              {busy ? "Generating…" : "Generate"}
            </button>
            <button className="btn ghost" disabled={busy || !res} onClick={generate}>
              Regenerate
            </button>
          </div>
          {busy && <div style={{ marginTop: 12 }}><Spinner label="one local model call…" /></div>}
          {error && <div style={{ marginTop: 12 }}><Err>{error}</Err></div>}
        </Card>

        <Card title="Draft" right={res ? <StatusPill status={v?.status} /> : undefined}>
          {!res && <p className="muted">No draft yet.</p>}
          {res && (
            <>
              <div className="answer" style={{ background: "var(--panel-2)", padding: 14,
                borderRadius: 8, border: "1px solid var(--line)" }}>
                {res.text}
              </div>
              <div className="btn-row" style={{ marginTop: 12 }}>
                <button className="btn ghost" onClick={copy}>
                  {copied ? "Copied" : "Copy to clipboard"}
                </button>
                <input value={instruction} onChange={(e) => setInstruction(e.target.value)}
                  placeholder="Improve: e.g. 'make the opening sharper'" style={{ flex: 2 }} />
                <button className="btn ghost" disabled={busy} onClick={improve}>Improve</button>
              </div>

              <div className="metrics" style={{ marginTop: 16 }}>
                <Metric k="Style match" v={`${Math.round((v?.style_match ?? 0) * 100)}%`} />
                <Metric k="Technical density" v={String(v?.stats?.technical_density ?? "—")} />
                <Metric k="Sources used" v={res.metrics.source_count} />
                <Metric k="LLM calls" v={res.metrics.llm_calls} />
                <Metric k="Tokens" v={res.metrics.tokens} />
                <Metric k="Latency" v={fmtMs(res.metrics.latency_ms)} />
              </div>
              <div style={{ marginTop: 10 }}><Bar value={v?.style_match ?? 0} /></div>

              {v?.issues?.length ? (
                <div style={{ marginTop: 12 }}>
                  <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                    Validator findings
                  </div>
                  <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
                    {v.issues.map((i) => <li key={i}>{i}</li>)}
                  </ul>
                </div>
              ) : (
                <p className="muted" style={{ fontSize: 12.5, marginTop: 12 }}>
                  All style checks passed{res.metrics.repaired ? " (after one repair call)" : ""}.
                </p>
              )}

              <div style={{ marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--line)" }}>
                <div className="muted" style={{ fontSize: 12, marginBottom: 6 }}>
                  Feedback — routed to the layer that owns the fix, not appended to the prompt
                </div>
                <div className="btn-row" style={{ flexWrap: "wrap", gap: 6 }}>
                  {["Too generic.", "Not technical enough.", "Doesn't sound like me.",
                    "The hook is weak.", "Too long.", "The CTA is wrong.",
                    "It keeps using generic AI buzzwords.", "Wrong audience."].map((f) => (
                    <button key={f} className="btn ghost" style={{ fontSize: 12 }}
                      disabled={fbBusy} onClick={() => sendFeedback(f)}>{f}</button>
                  ))}
                </div>
                <div className="btn-row" style={{ marginTop: 8 }}>
                  <input value={feedback} onChange={(e) => setFeedback(e.target.value)}
                    placeholder="…or describe it in your own words" style={{ flex: 3 }} />
                  <button className="btn ghost" disabled={fbBusy || !feedback.trim()}
                    onClick={() => sendFeedback(feedback)}>Classify</button>
                </div>
                {fbBusy && <div style={{ marginTop: 8 }}><Spinner label="classifying…" /></div>}
                {fbResult && (
                  <div style={{ marginTop: 10, background: "var(--panel-2)", padding: 12,
                    borderRadius: 8, border: "1px solid var(--line)" }}>
                    <div style={{ fontSize: 12.5, marginBottom: 6 }}>
                      <span className="pill ok">{fbResult.primary_layer}</span>{" "}
                      <span className="muted">
                        confidence {fbResult.confidence} ·{" "}
                        {fbResult.deterministic_fix_available
                          ? "fixable without the model"
                          : "may need model settings"}
                      </span>
                    </div>
                    {fbResult.classifications?.map((c: any, i: number) => (
                      <div key={i} style={{ fontSize: 12.5, marginBottom: 6 }}>
                        <div className="mono muted">{c.component}</div>
                        <div>{c.action}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <details style={{ marginTop: 14 }}>
                <summary>Style evidence &amp; checks</summary>
                <p className="mono muted" style={{ marginTop: 6 }}>{res.style_brief}</p>
                <div className="scroll-x">
                  <table>
                    <thead><tr><th>Check</th><th>Result</th></tr></thead>
                    <tbody>
                      {Object.entries(v?.checks || {}).map(([k, ok]) => (
                        <tr key={k}><td>{k}</td>
                          <td>{ok ? <span className="pill ok">PASS</span>
                            : <span className="pill bad">FAIL</span>}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div style={{ marginTop: 10 }}>
                  {res.examples.map((e) => (
                    <div key={e.doc_id} className="mono" style={{ marginBottom: 6 }}>
                      <span className="muted">[{e.created_at} · score {e.score}]</span> {e.preview}…
                    </div>
                  ))}
                </div>
              </details>
            </>
          )}
        </Card>
      </div>
    </>
  );
}
