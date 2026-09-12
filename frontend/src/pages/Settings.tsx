import { useEffect, useState } from "react";
import { api } from "../services/api";
import { Card, Err, Spinner } from "../components/Bits";

export default function Settings() {
  const [s, setS] = useState<any>(null);
  const [saved, setSaved] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.getSettings().then(setS).catch((e) => setError(String(e)));
  }, []);

  async function save(patch: Record<string, unknown>) {
    try {
      const r = await api.setSettings(patch);
      setS(r);
      setSaved(r.reload_needed ? "Saved — model reloads on next call." : "Saved.");
      setTimeout(() => setSaved(""), 2500);
    } catch (e) {
      setError(String(e));
    }
  }

  if (error) return <Err>{error}</Err>;
  if (!s) return <Spinner />;

  return (
    <>
      <div className="page-head">
        <h2>Settings</h2>
        <p>Runtime knobs. Changes apply immediately; model changes take effect on the next call.</p>
      </div>

      <Card title="Model">
        <div className="row">
          <div className="field">
            <label>Model preference</label>
            <select value={s.llm.prefer_size}
              onChange={(e) => save({ prefer_size: e.target.value })}>
              <option value="auto">Auto (largest that fits free RAM)</option>
              <option value="small">Small (faster)</option>
              <option value="large">Large (better quality, slower)</option>
            </select>
          </div>
          <div className="field">
            <label>Context window</label>
            <select value={s.llm.n_ctx} onChange={(e) => save({ n_ctx: Number(e.target.value) })}>
              <option value={2048}>2048</option>
              <option value={4096}>4096</option>
              <option value={8192}>8192</option>
            </select>
          </div>
        </div>
        <div className="row">
          <div className="field">
            <label>Temperature ({s.llm.temperature})</label>
            <input type="range" min={0} max={1.2} step={0.05} value={s.llm.temperature}
              onChange={(e) => save({ temperature: Number(e.target.value) })} />
          </div>
          <div className="field">
            <label>Max tokens</label>
            <input type="number" value={s.llm.max_tokens} min={64} max={2048} step={32}
              onChange={(e) => save({ max_tokens: Number(e.target.value) })} />
          </div>
        </div>
      </Card>

      <Card title="Harness">
        <div className="row">
          <div className="field">
            <label>Style examples per generation</label>
            <select value={s.harness.max_style_examples}
              onChange={(e) => save({ max_style_examples: Number(e.target.value) })}>
              <option value={1}>1 (cheapest)</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
              <option value={4}>4 (most context)</option>
            </select>
          </div>
          <div className="field">
            <label>Context budget</label>
            <input value={`${s.harness.max_context_chars} chars`} readOnly />
          </div>
        </div>
        <p className="muted" style={{ fontSize: 12.5 }}>
          Escalation thresholds — deterministic {s.harness.thresholds.deterministic} · NLP{" "}
          {s.harness.thresholds.nlp} · ML {s.harness.thresholds.ml} · retrieval{" "}
          {s.harness.thresholds.retrieval}. A processor below its threshold hands off to the
          next layer.
        </p>
      </Card>

      <Card title="Privacy">
        <p style={{ fontSize: 13, margin: 0 }}>
          The backend binds to 127.0.0.1 only. The LinkedIn export, the database, the models
          and every inference stay on this machine — there are no outbound API calls in the
          codebase, and the embedding model is loaded from the local HuggingFace cache with
          offline mode forced.
        </p>
      </Card>

      {saved && <div className="toast">{saved}</div>}
    </>
  );
}
