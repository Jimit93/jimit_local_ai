/**
 * Inline SVG charts for the evaluation panel.
 *
 * Deliberately dependency-free: a charting library would be the single largest
 * thing in the bundle, and these are three small static charts over data that is
 * already computed on the backend.
 */

const COMPONENTS = [
  ["content", "Content"],
  ["style", "Style"],
  ["structure", "Structure"],
  ["factual", "Factual"],
  ["technical", "Technical"],
  ["audience", "Audience"],
  ["efficiency", "Efficiency"],
] as const;

export function ScoreBars({ rows }: { rows: any[] }) {
  if (!rows?.length) return null;
  const a = rows[0];
  const b = rows[rows.length - 1];
  const W = 460;
  const rowH = 30;
  const labelW = 92;
  const barW = W - labelW - 54;

  return (
    <svg viewBox={`0 0 ${W} ${COMPONENTS.length * rowH + 34}`} width="100%"
      role="img" aria-label="Component scores, baseline versus final">
      <text x="0" y="12" fontSize="11" fill="var(--muted, #8b93a7)">
        Baseline (dim) vs final (bright) — 0 to 1
      </text>
      {COMPONENTS.map(([key, label], i) => {
        const av = a[`${key}_score`] ?? 0;
        const bv = b[`${key}_score`] ?? 0;
        const y = 30 + i * rowH;
        const better = bv >= av;
        return (
          <g key={key}>
            <text x="0" y={y + 13} fontSize="11.5" fill="currentColor">{label}</text>
            <rect x={labelW} y={y} width={barW} height="8" rx="4"
              fill="var(--line, #262b3a)" />
            <rect x={labelW} y={y} width={barW * av} height="8" rx="4"
              fill="#4b5772" />
            <rect x={labelW} y={y + 11} width={barW} height="8" rx="4"
              fill="var(--line, #262b3a)" />
            <rect x={labelW} y={y + 11} width={barW * bv} height="8" rx="4"
              fill={better ? "#4ade80" : "#f87171"} />
            <text x={labelW + barW + 8} y={y + 15} fontSize="11" fill="currentColor">
              {av.toFixed(2)}→{bv.toFixed(2)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export function IterationLine({ rows }: { rows: any[] }) {
  if (!rows?.length) return null;
  const W = 460;
  const H = 150;
  const padL = 34;
  const padB = 26;
  const n = rows.length;
  const x = (i: number) => padL + (i * (W - padL - 12)) / Math.max(1, n - 1);
  const y = (v: number) => 14 + (1 - v) * (H - padB - 14);
  const pts = rows.map((r, i) => `${x(i)},${y(r.overall_score)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label="Overall score by iteration">
      {[0, 0.5, 1].map((g) => (
        <g key={g}>
          <line x1={padL} y1={y(g)} x2={W - 12} y2={y(g)} stroke="var(--line, #262b3a)" />
          <text x="0" y={y(g) + 4} fontSize="10" fill="var(--muted, #8b93a7)">
            {g.toFixed(1)}
          </text>
        </g>
      ))}
      <polyline points={pts} fill="none" stroke="#60a5fa" strokeWidth="2" />
      {rows.map((r, i) => (
        <g key={r.run_id}>
          <circle cx={x(i)} cy={y(r.overall_score)} r="4" fill="#60a5fa" />
          <text x={x(i)} y={y(r.overall_score) - 9} fontSize="10" textAnchor="middle"
            fill="currentColor">{r.overall_score.toFixed(3)}</text>
          <text x={x(i)} y={H - 8} fontSize="10" textAnchor="middle"
            fill="var(--muted, #8b93a7)">{i}</text>
        </g>
      ))}
    </svg>
  );
}

export function EfficiencyBars({ rows }: { rows: any[] }) {
  if (!rows?.length) return null;
  const W = 460;
  const groupH = 46;
  const labelW = 78;
  const barW = W - labelW - 66;
  const metrics = [
    ["llm_calls", "LLM calls", (v: number) => String(v)],
    ["tokens", "Tokens", (v: number) => String(v)],
    ["latency_ms", "Latency", (v: number) => `${(v / 1000).toFixed(1)}s`],
  ] as const;

  return (
    <svg viewBox={`0 0 ${W} ${metrics.length * groupH + 16}`} width="100%" role="img"
      aria-label="Cost per generation by iteration">
      {metrics.map(([key, label, fmt], mi) => {
        const vals = rows.map((r) => r[key] ?? 0);
        const max = Math.max(...vals, 1);
        const yTop = mi * groupH + 12;
        return (
          <g key={key}>
            <text x="0" y={yTop + 14} fontSize="11.5" fill="currentColor">{label}</text>
            {rows.map((r, i) => {
              const v = r[key] ?? 0;
              const h = 7;
              const y = yTop + i * (h + 2);
              // Iteration 2 is the one that regressed on cost: mark it.
              const worse = i > 0 && v > vals[0];
              return (
                <g key={r.run_id}>
                  <rect x={labelW} y={y} width={(barW * v) / max} height={h} rx="3"
                    fill={i === 0 ? "#4b5772" : worse ? "#f87171" : "#4ade80"} />
                  <text x={labelW + barW + 6} y={y + h} fontSize="9.5"
                    fill="var(--muted, #8b93a7)">{`i${i} ${fmt(v)}`}</text>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
