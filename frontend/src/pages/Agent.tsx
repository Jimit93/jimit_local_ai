import { useEffect, useMemo, useState } from "react";
import { Card, Err, Metric, Spinner } from "../components/Bits";
import { Decisions, DiffView, HarnessFlow, MethodPill, TaskStatus, Timeline } from "../components/AgentBits";
import { agentApi, type AgentEvent, type TaskState, usePoll, useWorkspace } from "../services/agentApi";

const ACTIVE = ["queued", "running"];
const SAMPLES = [
  "Calculate 345 × 67",
  "git status",
  "where is the function run_task defined",
  "run tests/agent/test_agent_policy.py",
  "explain how the Governor class decides to defer heavy work",
  "summarize the architecture of this project",
];

function newSession() {
  const id = Math.random().toString(36).slice(2, 12);
  try { localStorage.setItem("jla.session", id); } catch { /* ignore */ }
  return id;
}
function currentSession() {
  try { return localStorage.getItem("jla.session") || newSession(); } catch { return newSession(); }
}

export default function Agent() {
  const [workspace, setWorkspace] = useWorkspace();
  const [sessionId, setSessionId] = useState(currentSession);
  const [input, setInput] = useState("");
  const [allowEdits, setAllowEdits] = useState(true);
  const [deepReview, setDeepReview] = useState(false);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsList = usePoll(() => agentApi.workspaces(), 0);
  useEffect(() => {
    if (!workspace && wsList.data?.workspaces?.length) setWorkspace(wsList.data.workspaces[0]);
  }, [wsList.data, workspace, setWorkspace]);

  const status = usePoll(() => agentApi.status(), 3000);
  const task = usePoll<TaskState | null>(
    () => (taskId ? agentApi.task(taskId) : Promise.resolve(null)), 1000, [taskId]);
  const t = task.data;
  const running = !!t && ACTIVE.includes(t.status);

  // Incremental event stream while a task is selected.
  useEffect(() => {
    if (!taskId) return;
    let alive = true;
    let after = 0;
    setEvents([]);
    const pull = async () => {
      try {
        const { events: ev } = await agentApi.events(taskId, after);
        if (!alive) return;
        if (ev.length) {
          after = ev[ev.length - 1].seq;
          setEvents((old) => [...old, ...ev]);
        }
      } catch { /* shown via task poll error */ }
    };
    pull();
    const id = window.setInterval(pull, 1000);
    return () => { alive = false; window.clearInterval(id); };
  }, [taskId]);

  // The transcript reloads when the session changes or the selected task changes state.
  const transcript = usePoll(() => agentApi.session(sessionId), 0, [sessionId, taskId, t?.status]);

  async function run(text?: string) {
    const request = (text ?? input).trim();
    if (!request || busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await agentApi.run({ request, workspace: workspace || undefined, session_id: sessionId,
        allow_edits: allowEdits, deep_review: deepReview });
      setInput("");
      setTaskId(r.task_id);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const gov = status.data?.governor;
  const step = t?.plan.find((s) => s.status === "running") || null;
  const lastMethod = step?.method || [...(t?.plan || [])].reverse().find((s) => s.method)?.method || null;
  const tools = useMemo(() => Array.from(new Set(events.filter((e) => e.kind === "step_end" && e.payload?.method === "deterministic")
    .map((e) => e.step))).slice(-4), [events]);
  const ctx = t?.data?.context_acct;
  const pending = t?.data?.pending_approval;
  const messages = transcript.data?.messages || [];

  return (
    <>
      <div className="page-head" style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h2>Coding Agent</h2>
          <p>Routed cheapest-first: tools → NLP → memory → small code model → capable code model. Every edit is checkpointed and verified.</p>
        </div>
        <div className="btn-row">
          <select aria-label="Workspace" value={workspace} onChange={(e) => setWorkspace(e.target.value)} style={{ width: 280 }}>
            {(wsList.data?.workspaces || []).map((w) => <option key={w} value={w}>{w}</option>)}
          </select>
          <button className="btn ghost" onClick={() => { setSessionId(newSession()); setTaskId(null); }}>New session</button>
        </div>
      </div>

      <div className="agent-grid">
        {/* ------------------------------------------------------------ chat */}
        <div>
          <Card title="Request">
            <div className="field">
              <textarea rows={4} value={input} aria-label="Coding request"
                placeholder="e.g. fix the failing test in tests/test_calc.py · where is build_context used · review my changes"
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); }} />
            </div>
            <div className="btn-row">
              <button className="btn" disabled={busy || !input.trim()} onClick={() => run()}>
                {busy ? "Submitting…" : "Run"}
              </button>
              <label className="check"><input type="checkbox" checked={allowEdits} onChange={(e) => setAllowEdits(e.target.checked)} /> Allow edits</label>
              <label className="check"><input type="checkbox" checked={deepReview} onChange={(e) => setDeepReview(e.target.checked)} /> Deep review</label>
              <span className="muted mono">Ctrl+Enter</span>
            </div>
            <div className="btn-row" style={{ marginTop: 10 }}>
              {SAMPLES.map((s) => (
                <button key={s} className="btn ghost chip" disabled={busy} onClick={() => run(s)}>{s}</button>
              ))}
            </div>
            {error && <div style={{ marginTop: 10 }}><Err>{error}</Err></div>}
          </Card>

          <Card title={`Session ${sessionId}`}>
            {!messages.length && <div className="muted">No messages yet. Ask something above.</div>}
            <div className="transcript">
              {messages.map((m: any) => (
                <div key={m.id} className={`msg ${m.role}`} onClick={() => m.task_id && setTaskId(m.task_id)}
                  role={m.task_id ? "button" : undefined} title={m.task_id ? "Show this task" : undefined}>
                  <div className="who">{m.role === "user" ? "You" : "Agent"} · {new Date(m.ts * 1000).toLocaleTimeString()}
                    {m.task_id === taskId && <span className="pill nlp" style={{ marginLeft: 6 }}>selected</span>}</div>
                  <div className="answer">{m.text.length > 600 ? m.text.slice(0, 600) + " …" : m.text}</div>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* ---------------------------------------------------------- status */}
        <div>
          <Card title="Agent status" right={gov && <span className={`pill ${gov.state === "NORMAL" ? "ok" : gov.state === "CAUTION" ? "warn" : "bad"}`}>
            {gov.state === "NORMAL" ? "✓" : "!"} CPU {gov.state}</span>}>
            <div className="metrics">
              <Metric k="Task" v={t ? <TaskStatus status={t.status} /> : <span className="muted">—</span>} small />
              <Metric k="Step" v={step?.name || t?.current_step || "—"} small />
              <Metric k="Method" v={lastMethod ? <MethodPill method={lastMethod} /> : "—"} small />
              <Metric k="CPU" v={gov?.last ? `${gov.last.sys_cpu}%` : "—"} />
              <Metric k="RAM free" v={gov?.last ? `${gov.last.ram_avail_gb} GB` : "—"} />
              <Metric k="Model" v={t?.data?.answer_model || t?.data?.proposal?.model || (status.data?.llm?.general_loaded ? status.data.llm.general_model : "idle")} small />
              <Metric k="Tools" v={tools.length ? tools.join(", ") : "—"} small />
              <Metric k="Context" v={ctx ? `${ctx.tokens_est} tok` : "—"} />
              <Metric k="Tokens" v={t?.counters?.tokens ?? 0} />
              <Metric k="LLM calls" v={t?.counters?.llm_calls ?? 0} />
            </div>
            {status.error && <div style={{ marginTop: 8 }}><Err>backend unreachable: {status.error}</Err></div>}
          </Card>

          {pending && t?.status === "awaiting_approval" && (
            <div className="approval">
              <b>Approval needed:</b> {pending.action} on <span className="mono">{pending.target}</span>
              <div className="muted">{pending.reason}. The change is checkpointed and can be undone.</div>
              <div className="btn-row" style={{ marginTop: 8 }}>
                <button className="btn" onClick={() => agentApi.approve(t.id, true)}>Approve</button>
                <button className="btn ghost" onClick={() => agentApi.approve(t.id, false)}>Deny</button>
              </div>
            </div>
          )}

          {t && (
            <Card title="Workflow" right={
              <div className="btn-row">
                {running && <Spinner label="working" />}
                {running && <button className="btn ghost chip" onClick={() => agentApi.cancel(t.id)}>Cancel</button>}
                {t.status === "interrupted" && <button className="btn chip" onClick={() => agentApi.resume(t.id)}>Resume</button>}
              </div>}>
              <div className="muted" style={{ marginBottom: 8 }}>{t.topic}</div>
              <Timeline task={t} />
              {t.resume_plan && <div className="muted" style={{ marginTop: 8 }}>Resume plan: {t.resume_plan.reason}</div>}
            </Card>
          )}

          <Card title="Harness view">
            <HarnessFlow task={t} events={events} />
          </Card>

          {t && !running && (
            <Card title="Result" right={t.last_checkpoint && (
              <button className="btn ghost chip" onClick={async () => {
                const r = await agentApi.rollback(t.last_checkpoint!);
                alert(r.ok ? "Rolled back." : `Not rolled back: ${JSON.stringify(r.conflicts)}`);
              }}>Undo change</button>)}>
              {t.data?.last_diff ? (
                <>
                  <div className="answer" style={{ marginBottom: 8 }}>{t.last_output.split("\n\n")[0]}</div>
                  <DiffView diff={t.data.last_diff} />
                </>
              ) : <div className="answer mono result-box">{t.last_output || "—"}</div>}
              {t.evaluation && <PredictedActual ev={t.evaluation} />}
            </Card>
          )}

          {t && <Card title="Routing decisions"><Decisions events={events} /></Card>}
          {task.error && <Err>{task.error}</Err>}
        </div>
      </div>
    </>
  );
}

function PredictedActual({ ev }: { ev: NonNullable<TaskState["evaluation"]> }) {
  const p = ev.predicted, a = ev.actual;
  const rows: [string, string, string, boolean | null][] = [
    ["Files changed", (p.files_changed || []).join(", ") || "none", (a.files_changed || []).join(", ") || "none",
      ev.checks.files_f1 === undefined ? null : ev.checks.files_f1 >= 0.99],
    ["Tests pass", p.tests_pass === null || p.tests_pass === undefined ? "n/a" : String(p.tests_pass),
      a.tests_pass === null || a.tests_pass === undefined ? "n/a" : String(a.tests_pass), ev.checks.tests_match ?? null],
    ["LLM calls", `≤ ${p.llm_calls_max}`, String(a.llm_calls), ev.checks.llm_budget_respected ?? null],
  ];
  return (
    <details style={{ marginTop: 12 }} open>
      <summary>Predicted vs actual · match {ev.match_score ?? "—"}</summary>
      <table>
        <thead><tr><th>Measure</th><th>Predicted</th><th>Actual</th><th>Match</th></tr></thead>
        <tbody>
          {rows.map(([k, pv, av, ok]) => (
            <tr key={k}><td>{k}</td><td className="mono">{pv}</td><td className="mono">{av}</td>
              <td>{ok === null ? "—" : ok ? <span className="pill ok">✓ yes</span> : <span className="pill bad">✕ no</span>}</td></tr>
          ))}
        </tbody>
      </table>
      <div className="muted mono" style={{ marginTop: 6 }}>
        {[
          a.duration_s !== undefined && `${a.duration_s}s`,
          a.iterations !== undefined && `${a.iterations} iterations`,
          a.repairs !== undefined && `${a.repairs} repairs`,
          a.tool_calls !== undefined && `${a.tool_calls} tool calls`,
          a.cache_hits ? `${a.cache_hits} cache hits` : null,
          a.processor && `answered by ${a.processor}`,
          a.cpu_peak !== undefined && `CPU avg ${a.cpu_avg}% peak ${a.cpu_peak}%`,
          a.context_tokens_est && `context ~${a.context_tokens_est} tokens`,
        ].filter(Boolean).join(" · ")}
      </div>
    </details>
  );
}
