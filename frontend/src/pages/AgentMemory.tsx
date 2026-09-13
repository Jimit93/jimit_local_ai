import { useState } from "react";
import { Card, Err, Metric } from "../components/Bits";
import { DiffView, TaskStatus } from "../components/AgentBits";
import { agentApi, usePoll, useWorkspace } from "../services/agentApi";

export default function AgentMemory() {
  const [workspace] = useWorkspace();
  const [q, setQ] = useState("");
  const [decision, setDecision] = useState("");
  const [cpDiff, setCpDiff] = useState<{ id: string; diff: string } | null>(null);
  const [analysis, setAnalysis] = useState<Record<string, any>>({});
  const [msg, setMsg] = useState<string | null>(null);
  const [allow, setAllow] = useState("");

  const mem = usePoll(() => agentApi.memory(workspace || undefined, q || undefined), 10000, [workspace, q]);
  const cps = usePoll(() => agentApi.checkpoints({ workspace: workspace || undefined }), 10000, [workspace]);
  const audit = usePoll(() => agentApi.audit(60), 10000);
  const ram = usePoll(() => agentApi.ram(), 8000);
  const p = mem.data?.project || {};
  const s = p.summary || {};

  async function act(fn: () => Promise<any>, ok: (r: any) => string) {
    setMsg(null);
    try { setMsg(ok(await fn())); mem.setData(null); } catch (e) { setMsg(String(e)); }
  }

  return (
    <>
      <div className="page-head">
        <h2>Memory &amp; checkpoints</h2>
        <p>What the agent remembers about this project, where unfinished work stopped, and every restorable checkpoint.</p>
      </div>
      {msg && <div className="toast" role="status" onClick={() => setMsg(null)}>{msg}</div>}

      <div className="grid-2">
        <Card title="Project memory">
          {mem.error && <Err>{mem.error}</Err>}
          <div className="metrics">
            <Metric k="Files" v={s.files ?? "—"} />
            <Metric k="Test command" v={p.test_command || "none"} small />
            <Metric k="Languages" v={Object.keys(s.languages || {}).slice(0, 3).join(", ") || "—"} small />
          </div>
          {p.important_files?.length > 0 && <div style={{ marginTop: 10 }}><label>Most-connected files</label>
            <div className="mono">{p.important_files.slice(0, 8).join(" · ")}</div></div>}
          {p.conventions?.length > 0 && <details style={{ marginTop: 10 }}><summary>{p.conventions.length} conventions (from CLAUDE.md/CONTRIBUTING)</summary>
            <ul>{p.conventions.map((c: string) => <li key={c}>{c}</li>)}</ul></details>}
          <div className="field" style={{ marginTop: 12 }}>
            <label>Search memory (BM25, no model)</label>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. rollback, tests, router" />
          </div>
          {(mem.data?.search || []).map((h: any) => (
            <div key={h.key} className="list-row"><span className="pill ml">{h.kind}</span>{" "}
              <span className="mono">{typeof h.value === "string" ? h.value : (h.value.text || h.value.signature || JSON.stringify(h.value)).slice(0, 160)}</span></div>
          ))}
        </Card>

        <Card title="Decisions & known issues">
          <div className="btn-row" style={{ marginBottom: 10 }}>
            <input aria-label="New decision" value={decision} onChange={(e) => setDecision(e.target.value)}
              placeholder="Record a decision the agent must respect" style={{ flex: 1 }} />
            <button className="btn ghost" disabled={!decision.trim()} onClick={() => act(
              () => agentApi.addDecision(decision.trim(), workspace), () => { setDecision(""); return "Decision saved."; })}>Save</button>
          </div>
          {(p.decisions || []).slice(0, 12).map((d: any, i: number) => (
            <div key={i} className="list-row"><span className="muted mono">{d.ts ? new Date(d.ts * 1000).toLocaleString() : ""}</span> {d.text}</div>
          ))}
          {(p.known_issues || []).length > 0 && <label style={{ marginTop: 10 }}>Known errors</label>}
          {(p.known_issues || []).slice(0, 8).map((k: any, i: number) => (
            <div key={i} className="list-row"><span className="pill bad">{k.count}×</span> <span className="mono">{String(k.signature).slice(13, 120)}</span>
              {k.resolution && <span className="muted"> → {k.resolution}</span>}</div>
          ))}
        </Card>
      </div>

      <Card title="Unfinished tasks (resume after restart)">
        {!(mem.data?.unfinished || []).length && <div className="muted">No unfinished tasks.</div>}
        <table>
          <tbody>
            {(mem.data?.unfinished || []).map((t: any) => (
              <tr key={t.id}>
                <td><TaskStatus status={t.status} /></td>
                <td>{t.topic}<div className="muted mono">step {t.current_step || "—"} · resumed {t.resume_count}×</div>
                  {analysis[t.id] && <div className="muted">{analysis[t.id].reason}
                    {analysis[t.id].checks && <span className="mono"> [{analysis[t.id].checks.map((c: any) => `${c.ok ? "✓" : "✕"} ${c.check}`).join(" · ")}]</span>}</div>}</td>
                <td className="num" style={{ whiteSpace: "nowrap" }}>
                  <button className="btn ghost chip" onClick={async () => {
                    const a = await agentApi.resumeAnalysis(t.id);
                    setAnalysis((o) => ({ ...o, [t.id]: a }));
                  }}>Check</button>{" "}
                  <button className="btn chip" onClick={() => act(() => agentApi.resume(t.id),
                    (r) => r.ok ? `Resuming at ${r.resume_step}` : `Not resumable: ${r.reason}`)}>Resume</button>{" "}
                  <button className="btn ghost chip" onClick={() => act(() => agentApi.cancel(t.id), () => "Cancelled.")}>Cancel</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title="Checkpoints">
        <div className="scroll-x">
          <table>
            <thead><tr><th>When</th><th>Label</th><th>Files</th><th /></tr></thead>
            <tbody>
              {(cps.data?.checkpoints || []).slice(0, 30).map((c: any) => (
                <tr key={c.id}>
                  <td className="mono">{new Date(c.ts * 1000).toLocaleString()}</td>
                  <td>{c.label}{c.restored_at && <span className="pill ok" style={{ marginLeft: 6 }}>✓ restored</span>}</td>
                  <td className="mono">{c.files.map((f: any) => f.path).join(", ")}</td>
                  <td className="num" style={{ whiteSpace: "nowrap" }}>
                    <button className="btn ghost chip" onClick={async () => setCpDiff({ id: c.id, diff: (await agentApi.checkpointDiff(c.id)).diff })}>Diff</button>{" "}
                    <button className="btn ghost chip" onClick={() => act(() => agentApi.rollback(c.id),
                      (r) => r.ok ? `Restored ${r.actions.length} file(s).` : `Conflicts, nothing overwritten: ${r.conflicts.map((x: any) => x.path).join(", ")}`)}>Rollback</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {cpDiff && <div style={{ marginTop: 10 }}><label>Changes since checkpoint {cpDiff.id}</label><DiffView diff={cpDiff.diff} /></div>}
      </Card>

      <div className="grid-2">
        <Card title={`RAM · ${ram.data?.available_gb ?? "—"} GB free`}>
          <div className="muted" style={{ marginBottom: 8 }}>
            Before loading a model the agent frees RAM automatically only from its own resources and from apps on your allowlist.
            Everything else needs your approval here. System and security processes are never touched.
          </div>
          <div className="btn-row" style={{ marginBottom: 10 }}>
            <input aria-label="Allowlist" value={allow} placeholder={(ram.data?.allowlist || []).join("; ") || "e.g. LM Studio.exe; Teams.exe"}
              onChange={(e) => setAllow(e.target.value)} style={{ flex: 1 }} />
            <button className="btn ghost" onClick={() => act(() => agentApi.setAllowlist(allow.split(/[;,]/)),
              (r) => `Allowlist: ${r.allowlist.join(", ") || "empty"}`)}>Save allowlist</button>
          </div>
          <table>
            <thead><tr><th>Process</th><th className="num">RAM</th><th>Category</th><th /></tr></thead>
            <tbody>
              {(ram.data?.processes || []).slice(0, 14).map((pr: any) => (
                <tr key={pr.pid}>
                  <td className="mono">{pr.name} <span className="muted">#{pr.pid}</span></td>
                  <td className="num">{pr.rss_mb} MB</td>
                  <td><span className={`pill ${pr.category === "protected" ? "ok" : pr.category === "candidate" ? "warn" : "nlp"}`}>{pr.category}</span></td>
                  <td className="num">{pr.category !== "protected" && (
                    <button className="btn ghost chip" onClick={() => {
                      if (!confirm(`Close ${pr.name} (#${pr.pid})? Unsaved work in that app may be lost.`)) return;
                      act(() => agentApi.terminate(pr.pid, pr.create_time, pr.name),
                        (r) => r.ok ? `${r.name} ${r.how}; ${r.available_gb_after} GB free` : `Not closed: ${r.reason}`);
                    }}>Close</button>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Audit log">
          <div className="audit">
            {(audit.data?.audit || []).map((a: any) => (
              <div key={a.id} className="list-row mono">
                <span className={`pill ${a.decision === "allow" || a.decision === "approved" ? "ok" : a.decision === "deny" ? "bad" : "warn"}`}>{a.decision}</span>{" "}
                {new Date(a.ts * 1000).toLocaleTimeString()} {a.action} <span className="muted">{String(a.target || "").slice(0, 70)}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </>
  );
}
