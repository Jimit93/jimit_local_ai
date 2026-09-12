import type { ReactNode } from "react";

export function Card({ title, children, right }: {
  title?: string; children: ReactNode; right?: ReactNode;
}) {
  return (
    <div className="card">
      {(title || right) && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          {title && <h3>{title}</h3>}
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

export function Metric({ k, v, small }: { k: string; v: ReactNode; small?: boolean }) {
  return (
    <div className="metric">
      <div className="k">{k}</div>
      <div className={small ? "v small" : "v"}>{v}</div>
    </div>
  );
}

const LAYERS = ["deterministic", "nlp", "ml", "retrieval", "llm", "hybrid"];

export function LayerPill({ layer }: { layer?: string | null }) {
  const l = (layer || "").toLowerCase();
  const cls = LAYERS.includes(l) ? l : "nlp";
  return <span className={`pill ${cls}`}>{(layer || "?").toUpperCase()}</span>;
}

export function StatusPill({ status }: { status?: string }) {
  const s = (status || "").toUpperCase();
  const cls = s === "SUCCESS" ? "ok" : s === "FAILURE" ? "bad" : "warn";
  return <span className={`pill ${cls}`}>{s || "—"}</span>;
}

export function Bar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const color = pct >= 85 ? "var(--ok)" : pct >= 60 ? "var(--warn)" : "var(--bad)";
  return <div className="bar"><i style={{ width: `${pct}%`, background: color }} /></div>;
}

export function Err({ children }: { children: ReactNode }) {
  return <div className="err">{children}</div>;
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="muted" style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
      <span className="spinner" /> {label || "working…"}
    </span>
  );
}

export function fmtMs(ms?: number | null) {
  if (ms === undefined || ms === null) return "—";
  return ms < 1000 ? `${ms.toFixed(ms < 10 ? 1 : 0)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
