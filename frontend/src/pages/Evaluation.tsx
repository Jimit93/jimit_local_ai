import { useEffect, useState } from "react";
import { api } from "../services/api";
import { Card, Err, Metric, Spinner } from "../components/Bits";
import { EfficiencyBars, IterationLine, ScoreBars } from "../components/EvalGraphs";

const COMPONENTS = [
  ["content", "Content"],
  ["style", "Style"],
  ["structure", "Structure"],
  ["factual", "Factual"],
  ["technical", "Technical"],
  ["audience", "Audience"],
  ["efficiency", "Efficiency"],
] as const;

function blocks(v: number) {
  const filled = Math.round(Math.max(0, Math.min(1, v)) * 10);
  return "█".repeat(filled) + "░".repeat(10 - filled);
}

const PROCESSOR_PATH = [
  "Style engine (measured profile)",
  "Retrieval (BM25 + MiniLM/FAISS, chunk → document)",
  "Content plan (deterministic)",
  "Prompt builder",
  "Qwen (one call)",
  "Validator (deterministic QC)",
];

const BT_ROWS = [
  ["pass_rate", "Pass rate (SUCCESS)", "pct"],
  ["complete_ending_rate", "Complete ending", "pct"],
  ["hashtag_rate", "Has hashtags", "pct"],
  ["mean_overall", "Mean overall score", "num"],
  ["mean_critical_failures", "Mean critical failures", "num"],
  ["posts_with_banned_phrases", "Posts with banned phrases", "int"],
  ["mean_llm_calls", "Mean model calls", "num"],
  ["mean_tokens", "Mean tokens", "int"],
  ["mean_latency_ms", "Mean latency", "ms"],
] as const;

function fmtBt(v: any, kind: string) {
  if (v === undefined || v === null) return "—";
  if (kind === "pct") return `${Math.round(v * 100)}%`;
  if (kind === "ms") return `${(v / 1000).toFixed(1)}s`;
  if (kind === "int") return String(Math.round(v));
  return Number(v).toFixed(3);
}

function RefineBacktestCard() {
  const [bt, setBt] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.refineBacktestLatest().then(setBt).catch(() => setBt(null));
  }, []);

  async function run() {
    setBusy(true);
    setErr("");
    try {
      setBt(await api.refineBacktest({}));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  const a = bt?.aggregate?.one_shot || {};
  const d = bt?.aggregate?.refine_draft || {};
  const r = bt?.aggregate?.refine || {};
  const curveRows = (bt?.iteration_curve || []).map((v: number, i: number) => ({
    run_id: `i${i}`, overall_score: v,
  }));

  return (
    <Card title="Refine loop backtest — Describe → Generate → Test → Refine">
      <div className="btn-row">
        <button className="btn" disabled={busy} onClick={run}>Run backtest</button>
        {busy && <Spinner label="one-shot vs refine on every brief — several minutes on CPU…" />}
      </div>
      {err && <div style={{ marginTop: 10 }}><Err>{err}</Err></div>}
      {!bt && !busy && (
        <p className="muted" style={{ marginTop: 10 }}>No backtest yet. It compares the normal
          one-shot writer against the refine loop on fixed briefs, same seeded model.</p>
      )}
      {bt && (
        <>
          <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
            Run <span className="mono">{bt.run_id}</span> · {bt.model} · seed {bt.seed} ·{" "}
            {bt.n_briefs} briefs · {bt.duration_s}s · up to {bt.settings.max_iterations} micro-iterations,{" "}
            {bt.settings.max_llm_fixes} model fixes
          </p>
          <div className="grid-2">
            <div className="scroll-x">
              <table>
                <thead><tr><th>Metric</th><th>One-shot</th><th>Refine draft</th><th>Refine loop</th></tr></thead>
                <tbody>
                  {BT_ROWS.map(([k, label, kind]) => (
                    <tr key={k}>
                      <td>{label}</td>
                      <td className="mono">{fmtBt(a[k], kind)}</td>
                      <td className="mono">{fmtBt(d[k], kind)}</td>
                      <td className="mono">{fmtBt(r[k], kind)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>
                Refine draft → Refine loop is paired (same draft, before and after fixes). One-shot is a
                different sample: the seed is set once when the model loads.
              </p>
            </div>
            <div>
              <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                Mean score by micro-iteration — last one that still helped:{" "}
                <strong>{bt.last_improving_iteration}</strong> of {bt.settings.max_iterations} allowed
              </div>
              <IterationLine rows={curveRows} />
            </div>
          </div>
          {bt.reviewer_findings?.length ? (
            <div style={{ marginTop: 12 }}>
              <div className="muted" style={{ fontSize: 12, marginBottom: 4 }}>
                What the checks could not see — read from the generated text
              </div>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5, lineHeight: 1.6 }}>
                {bt.reviewer_findings.map((f: string) => <li key={f}>{f}</li>)}
              </ul>
            </div>
          ) : null}
          <div className="scroll-x" style={{ marginTop: 12 }}>
            <table>
              <thead>
                <tr><th>Brief</th><th>One-shot</th><th>Refine loop</th><th>Stopped</th></tr>
              </thead>
              <tbody>
                {(bt.rows || []).map((row: any) => row.error ? (
                  <tr key={row.id}><td className="mono">{row.id}</td><td colSpan={3}>{row.error}</td></tr>
                ) : (
                  <tr key={row.id}>
                    <td className="mono" style={{ fontSize: 12 }}>{row.id}</td>
                    <td style={{ fontSize: 12 }}>
                      {row.one_shot.status} · {row.one_shot.overall.toFixed(3)} · {row.one_shot.llm_calls} calls
                      {row.one_shot.critical_failures.length > 0 &&
                        <div className="muted">{row.one_shot.critical_failures.join(", ")}</div>}
                    </td>
                    <td style={{ fontSize: 12 }}>
                      {row.refine.status} · {row.refine.overall.toFixed(3)} · {row.refine.llm_calls} calls ·{" "}
                      {row.refine.iterations} iters
                      {row.refine.critical_failures.length > 0 &&
                        <div className="muted">{row.refine.critical_failures.join(", ")}</div>}
                    </td>
                    <td className="mono" style={{ fontSize: 11.5 }}>{row.refine.stopped_because}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}

export default function Evaluation() {
  const [rep, setRep] = useState<any>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);

  useEffect(() => {
    api
      .evaluationReport()
      .then(setRep)
      .catch(() => setError("No evaluation has been run yet. Artifacts live in reports/evaluation/."))
      .finally(() => setBusy(false));
  }, []);

  const cmp = rep?.comparison;
  const rows: any[] = cmp?.all_iterations || [];
  const a = cmp?.version_a;
  const b = cmp?.version_b;
  const predicted = rep?.predicted;
  const iters: any[] = rep?.iterations?.iterations || [];

  return (
    <>
      <div className="page-head">
        <h2>Evaluation</h2>
        <p>
          Predicted profile versus actual output, scored deterministically. No model call is
          made to produce any number on this page.
        </p>
      </div>

      {busy && <Card><Spinner label="loading evaluation…" /></Card>}
      {error && <Card><Err>{error}</Err></Card>}

      {cmp && (
        <>
          <div className="grid-2">
            <Card title="Generated content — final">
              <div className="answer" style={{ background: "var(--panel-2)", padding: 14,
                borderRadius: 8, border: "1px solid var(--line)" }}>
                {b.output}
              </div>
              <details style={{ marginTop: 10 }}>
                <summary>Baseline output, for comparison</summary>
                <div className="answer muted" style={{ marginTop: 8, fontSize: 13 }}>
                  {a.output}
                </div>
              </details>
              <div className="metrics" style={{ marginTop: 14 }}>
                <Metric k="Overall" v={b.overall_score.toFixed(4)} />
                <Metric k="Baseline" v={a.overall_score.toFixed(4)} />
                <Metric k="Delta" v={`+${cmp.delta_b_minus_a.overall_score}`} />
                <Metric k="Validator" v={b.validator_status} />
                <Metric k="Facts cited" v={`${b.facts_cited.length} of 18`} />
                <Metric k="Inversion" v={b.audience_inversion ? "yes" : "none"} />
              </div>
            </Card>

            <Card title="Predicted profile (from the corpus)">
              {predicted ? (
                <table>
                  <tbody>
                    <tr><td>Audience</td><td>{predicted.audience}</td></tr>
                    <tr><td>Pillar</td><td>{predicted.content_pillar}</td></tr>
                    <tr><td>Expected length</td>
                      <td>{predicted.expected_length.min}–{predicted.expected_length.max} words
                        (corpus median {predicted.expected_length.corpus_median})</td></tr>
                    <tr><td>Expected hook</td>
                      <td>{predicted.hook.type} ({Math.round(predicted.hook.rate_in_corpus * 100)}% of posts)</td></tr>
                    <tr><td>Paragraphs</td>
                      <td>{predicted.paragraph_length_mean.target_paragraphs.join("–")}</td></tr>
                    <tr><td>Technical density floor</td>
                      <td>{predicted.technical_density.floor}</td></tr>
                    <tr><td>CTA expected</td>
                      <td>{predicted.cta_type.expected ? "yes" : `no (${Math.round(predicted.cta_type.usage_rate * 100)}% of posts)`}</td></tr>
                    <tr><td>Hashtags</td>
                      <td>{predicted.hashtag_count.min}–{predicted.hashtag_count.max}</td></tr>
                    <tr><td>Emoji policy</td><td>{predicted.emoji_policy}</td></tr>
                    <tr><td>Numbers available</td>
                      <td>{predicted.expected_numbers.length} (cite ≥{predicted.expected_number_count_min})</td></tr>
                  </tbody>
                </table>
              ) : <p className="muted">No predicted profile.</p>}
            </Card>
          </div>

          <Card title="Evaluation — baseline vs final">
            <div className="scroll-x">
              <table>
                <thead>
                  <tr><th>Component</th><th>Baseline</th><th>Final</th><th></th><th>Δ</th></tr>
                </thead>
                <tbody>
                  {COMPONENTS.map(([k, label]) => {
                    const av = a[`${k}_score`];
                    const bv = b[`${k}_score`];
                    const d = +(bv - av).toFixed(4);
                    return (
                      <tr key={k}>
                        <td>{label}</td>
                        <td className="mono">{av.toFixed(3)}</td>
                        <td className="mono">{bv.toFixed(3)}</td>
                        <td className="mono" style={{ letterSpacing: -1 }}>{blocks(bv)}</td>
                        <td className="mono" style={{
                          color: d > 0 ? "var(--ok)" : d < 0 ? "var(--bad)" : "inherit" }}>
                          {d > 0 ? `+${d}` : d}
                        </td>
                      </tr>
                    );
                  })}
                  <tr>
                    <td><strong>Overall</strong></td>
                    <td className="mono"><strong>{a.overall_score.toFixed(4)}</strong></td>
                    <td className="mono"><strong>{b.overall_score.toFixed(4)}</strong></td>
                    <td className="mono" style={{ letterSpacing: -1 }}>{blocks(b.overall_score)}</td>
                    <td className="mono" style={{ color: "var(--ok)" }}>
                      <strong>+{cmp.delta_b_minus_a.overall_score}</strong>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
              Weights: {Object.entries(cmp.weights).map(([k, v]) => `${k} ${v}`).join(" · ")}
            </p>
          </Card>

          <div className="grid-2">
            <Card title="Component scores">
              <ScoreBars rows={rows} />
            </Card>
            <Card title="Overall score by iteration">
              <IterationLine rows={rows} />
            </Card>
          </div>

          <div className="grid-2">
            <Card title="Cost per generation">
              <EfficiencyBars rows={rows} />
              <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                Iteration 2 in red: the validator gate fired the repair pass, which cost a
                second model call and changed nothing. Iteration 3 removed the need for it.
              </p>
            </Card>
            <Card title="Processor path">
              <ol style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.9 }}>
                {PROCESSOR_PATH.map((p) => <li key={p}>{p}</li>)}
              </ol>
              <div className="metrics" style={{ marginTop: 14 }}>
                <Metric k="LLM calls" v={b.llm_calls} />
                <Metric k="Tokens" v={b.tokens} />
                <Metric k="Latency" v={`${(b.latency_ms / 1000).toFixed(1)}s`} />
                <Metric k="Sources" v={b.retrieval_count} />
              </div>
            </Card>
          </div>

          <Card title="Iteration history">
            <div className="scroll-x">
              <table>
                <thead>
                  <tr><th>#</th><th>Change</th><th>Layer</th><th>Score</th><th>Regression</th></tr>
                </thead>
                <tbody>
                  {iters.map((it) => (
                    <tr key={it.iteration}>
                      <td className="mono">{it.iteration}</td>
                      <td>
                        <strong>{it.label}</strong>
                        {it.key_metric && <div className="muted" style={{ fontSize: 12 }}>{it.key_metric}</div>}
                        {it.tradeoff && <div className="muted" style={{ fontSize: 12 }}>⚠ {it.tradeoff}</div>}
                      </td>
                      <td className="mono" style={{ fontSize: 11.5 }}>{it.layer || "—"}</td>
                      <td className="mono">
                        {it.score_before !== null && it.score_before !== undefined
                          ? `${it.score_before} → ${it.score_after}` : it.score_after}
                      </td>
                      <td style={{ fontSize: 12 }}>{it.regression_result}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Caveats">
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, lineHeight: 1.7 }}>
              {(cmp.caveats || []).map((c: string) => <li key={c}>{c}</li>)}
            </ul>
          </Card>
        </>
      )}

      <RefineBacktestCard />
    </>
  );
}
