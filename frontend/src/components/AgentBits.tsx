import type { AgentEvent, Step, TaskState } from "../services/agentApi";

export const METHOD_LABEL: Record<string, string> = {
  deterministic: "TOOL", nlp: "NLP", memory: "MEMORY", ml: "ML", cache: "CACHE",
  llm_small: "LLM · small", llm_capable: "LLM · capable", llm_general: "LLM · general", skipped: "SKIPPED",
  retrieval: "RETRIEVAL", hybrid: "HYBRID", llm: "LLM",
};
const METHOD_CLASS: Record<string, string> = {
  deterministic: "deterministic", nlp: "nlp", memory: "ml", ml: "ml", cache: "ml", retrieval: "retrieval",
  llm_small: "llm", llm_capable: "llm", llm_general: "llm", llm: "llm", hybrid: "hybrid", skipped: "warn",
};

export function MethodPill({ method }: { method?: string | null }) {
  if (!method) return null;
  return <span className={`pill ${METHOD_CLASS[method] || "nlp"}`}>{METHOD_LABEL[method] || method.toUpperCase()}</span>;
}

export function TaskStatus({ status }: { status: string }) {
  const cls = status === "completed" ? "ok" : ["failed", "blocked", "cancelled"].includes(status) ? "bad" : "warn";
  const icon = status === "completed" ? "✓" : status === "failed" || status === "blocked" ? "✕" : status === "cancelled" ? "⦸" : "●";
  return <span className={`pill ${cls}`}>{icon} {status.replace("_", " ").toUpperCase()}</span>;
}

const ICON: Record<Step["status"], string> = { done: "✓", running: "●", pending: "○", failed: "✕", skipped: "–" };
const STEP_LABEL: Record<string, string> = {
  tool: "Run tool", harness: "Route through harness", run_tests: "Reproduce with tests", locate: "Find relevant files",
  context: "Gather minimal context", answer: "Answer", validate_answer: "Check answer against the code",
  propose_edit: "Propose change", apply_edit: "Apply change (checkpointed)", deterministic_fix: "Apply rename (no model)",
  verify_syntax: "Verify syntax", verify_lint: "Verify lint", verify_tests: "Run tests", diff: "Collect diff",
  static_review: "Automated review", llm_review: "Model review", report: "Report",
};

export function Timeline({ task }: { task: TaskState }) {
  return (
    <ol className="timeline">
      <li className="done"><span className="ic">✓</span><div><b>Understand request</b>
        <div className="muted">intent {task.intent} · {String(task.data?.intent?.stage ?? "")}</div></div></li>
      {task.plan.map((s) => (
        <li key={s.name} className={s.status}>
          <span className="ic">{ICON[s.status]}</span>
          <div style={{ minWidth: 0 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <b>{STEP_LABEL[s.kind] || s.name}</b>
              <MethodPill method={s.method} />
              {s.attempts > 1 && <span className="muted mono">attempt {s.attempts}</span>}
            </div>
            {s.summary && <div className="muted step-summary">{s.summary}</div>}
          </div>
        </li>
      ))}
      <li className={task.status === "completed" ? "done" : ["failed", "blocked", "cancelled"].includes(task.status) ? "failed" : "pending"}>
        <span className="ic">{task.status === "completed" ? "✓" : ["failed", "blocked", "cancelled"].includes(task.status) ? "✕" : "○"}</span>
        <div><b>Complete</b></div>
      </li>
    </ol>
  );
}

// ------------------------------------------------------------------ harness view

type Node = { id: string; label: string; x: number; y: number };
const NODES: Node[] = [
  { id: "request", label: "Request", x: 60, y: 90 },
  { id: "classifier", label: "Classifier", x: 170, y: 90 },
  { id: "tool", label: "Tool", x: 300, y: 24 },
  { id: "nlp", label: "NLP", x: 300, y: 68 },
  { id: "ml", label: "ML / memory", x: 300, y: 112 },
  { id: "llm", label: "LLM", x: 300, y: 156 },
  { id: "validator", label: "Validator", x: 430, y: 90 },
  { id: "evaluator", label: "Evaluator", x: 540, y: 90 },
  { id: "repair", label: "Repair", x: 485, y: 162 },
  { id: "result", label: "Result", x: 640, y: 90 },
];
const EDGES = [
  ["request", "classifier"], ["classifier", "tool"], ["classifier", "nlp"], ["classifier", "ml"], ["classifier", "llm"],
  ["tool", "validator"], ["nlp", "validator"], ["ml", "validator"], ["llm", "validator"], ["validator", "evaluator"],
  ["evaluator", "result"], ["validator", "repair"], ["repair", "llm"],
];

function nodeFor(kind: string, method: string | null): string {
  if (["apply_edit", "verify_syntax", "verify_lint", "verify_tests", "validate_answer"].includes(kind)) return "validator";
  if (kind === "report") return "evaluator";
  if (method?.startsWith("llm")) return "llm";
  if (method === "nlp" || kind === "context") return "nlp";
  if (method === "memory" || method === "cache" || method === "ml") return "ml";
  if (["answer", "propose_edit", "llm_review"].includes(kind)) return "llm";
  return "tool";
}

export function HarnessFlow({ task, events }: { task: TaskState | null; events: AgentEvent[] }) {
  const used = new Set<string>(["request"]);
  let active = "request";
  if (task) {
    used.add("classifier");
    active = "classifier";
    for (const s of task.plan) {
      if (s.status === "done" || s.status === "failed") used.add(nodeFor(s.kind, s.method));
      if (s.status === "running") active = nodeFor(s.kind, s.method);
    }
    if (events.some((e) => e.kind === "repair")) used.add("repair");
    const last = events[events.length - 1];
    if (last?.kind === "repair") active = "repair";
    if (task.status === "completed") { used.add("evaluator"); used.add("result"); active = "result"; }
    if (["failed", "blocked", "cancelled"].includes(task.status)) active = "";
  }
  const pos = Object.fromEntries(NODES.map((n) => [n.id, n]));
  return (
    <svg viewBox="0 0 700 190" width="100%" role="img" aria-label="Harness routing flow with the active stage highlighted">
      <defs>
        <marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" fill="#4a5568" />
        </marker>
      </defs>
      {EDGES.map(([a, b]) => {
        const A = pos[a], B = pos[b];
        const on = used.has(a) && used.has(b);
        return <line key={a + b} x1={A.x + 40} y1={A.y} x2={B.x - 42} y2={B.y} stroke={on ? "#4da3ff" : "#2c3444"}
          strokeWidth={on ? 1.6 : 1} markerEnd="url(#arr)" />;
      })}
      {NODES.map((n) => {
        const isActive = n.id === active;
        const isUsed = used.has(n.id);
        return (
          <g key={n.id}>
            <rect x={n.x - 42} y={n.y - 14} width={84} height={28} rx={8}
              fill={isActive ? "#1b3a5c" : "#171d28"} stroke={isActive ? "#4da3ff" : isUsed ? "#3a4a63" : "#232b39"}
              strokeWidth={isActive ? 2 : 1} />
            <text x={n.x} y={n.y + 4} fontSize="11" textAnchor="middle" fill={isUsed || isActive ? "#e6ecf3" : "#8a97a8"}>
              {isUsed && !isActive ? "✓ " : ""}{n.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function DiffView({ diff, max = 400 }: { diff: string; max?: number }) {
  if (!diff) return <div className="muted">No changes.</div>;
  const lines = diff.split("\n");
  return (
    <pre className="diff mono">
      {lines.slice(0, max).map((l, i) => (
        <div key={i} className={l.startsWith("+++") || l.startsWith("---") ? "meta" : l.startsWith("+") ? "add"
          : l.startsWith("-") ? "del" : l.startsWith("@@") ? "hunk" : ""}>{l || " "}</div>
      ))}
      {lines.length > max && <div className="muted">… {lines.length - max} more lines</div>}
    </pre>
  );
}

/** Routing explanations from the event stream: what was chosen and why (no chain-of-thought). */
export function Decisions({ events }: { events: AgentEvent[] }) {
  const rows = events.filter((e) => e.payload?.explanation || ["repair", "governor", "ram", "approval", "resume",
    "rollback", "stop"].includes(e.kind));
  if (!rows.length) return <div className="muted">No decisions yet.</div>;
  return (
    <ul className="decisions">
      {rows.slice(-14).map((e) => (
        <li key={e.seq}>
          <span className="mono muted">{new Date(e.ts * 1000).toLocaleTimeString()}</span>{" "}
          <b>{e.step ? `${e.step}: ` : ""}{e.kind.replace("_", " ")}</b>{" "}
          {e.method && <MethodPill method={e.method} />}{" "}
          <span>{e.payload.explanation || e.payload.reason || e.payload.error || ""}</span>
        </li>
      ))}
    </ul>
  );
}
