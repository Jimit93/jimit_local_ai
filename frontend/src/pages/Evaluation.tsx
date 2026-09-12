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
    </>
  );
}
