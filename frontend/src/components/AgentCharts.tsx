/**
 * Dependency-free SVG charts for the agent dashboards.
 *
 * Conventions: one y-axis per chart (different measures get different charts),
 * 2px lines, bars <= 24px with 4px rounded data-ends, hairline gridlines, labels in
 * text colors (never the series color), a hover tooltip on every plotted mark.
 * Categorical slots are the validated dark steps (checked against #131720).
 */
import { useMemo, useRef, useState, type ReactNode } from "react";

export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500"]; // validated order, never cycled
const GRID = "#232b39";
const AXIS = "#8a97a8";
const INK = "#e6ecf3";

function niceMax(v: number) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / p;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
}
const fmt = (v: number) =>
  Math.abs(v) >= 10000 ? `${(v / 1000).toFixed(0)}K` : Math.abs(v) >= 1000 ? v.toLocaleString() : `${+v.toFixed(2)}`;

function Tip({ x, y, children, width }: { x: number; y: number; children: ReactNode; width: number }) {
  const left = Math.min(Math.max(x + 12, 0), width - 170);
  return (
    <div className="viz-tip" style={{ left, top: Math.max(0, y - 10) }}>
      {children}
    </div>
  );
}

export interface LineSeries {
  name: string;
  points: { x: number; y: number }[];
}

/** Time series with crosshair tooltip and optional horizontal reference lines (e.g. the 70% cap). */
export function LineChart({
  series, height = 170, yMax, yLabel, refLines = [], xFormat = (v) => new Date(v * 1000).toLocaleTimeString(),
  ariaLabel,
}: {
  series: LineSeries[];
  height?: number;
  yMax?: number;
  yLabel?: string;
  refLines?: { y: number; label: string }[];
  xFormat?: (v: number) => string;
  ariaLabel: string;
}) {
  const W = 560;
  const H = height;
  const padL = 38, padR = 12, padT = 12, padB = 22;
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ i: number; px: number; py: number } | null>(null);
  const all = series.flatMap((s) => s.points);
  if (!all.length) return <div className="muted" style={{ fontSize: 12 }}>No samples yet.</div>;
  const xs = all.map((p) => p.x);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const ym = yMax ?? niceMax(Math.max(...all.map((p) => p.y), ...refLines.map((r) => r.y)));
  const sx = (v: number) => padL + ((v - x0) / Math.max(1e-9, x1 - x0)) * (W - padL - padR);
  const sy = (v: number) => padT + (1 - v / ym) * (H - padT - padB);
  const base = series[0].points;

  function onMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const vx = ((e.clientX - rect.left) / rect.width) * W;
    let best = 0;
    base.forEach((p, i) => {
      if (Math.abs(sx(p.x) - vx) < Math.abs(sx(base[best].x) - vx)) best = i;
    });
    setHover({ i: best, px: e.clientX - rect.left, py: e.clientY - rect.top });
  }

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={ariaLabel}
        onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line x1={padL} x2={W - padR} y1={sy(g * ym)} y2={sy(g * ym)} stroke={GRID} strokeWidth={1} />
            <text x={padL - 6} y={sy(g * ym) + 4} fontSize="10" fill={AXIS} textAnchor="end">{fmt(g * ym)}</text>
          </g>
        ))}
        {yLabel && <text x={padL} y={9} fontSize="10" fill={AXIS}>{yLabel}</text>}
        {refLines.map((r) => (
          <g key={r.label}>
            <line x1={padL} x2={W - padR} y1={sy(r.y)} y2={sy(r.y)} stroke={AXIS} strokeWidth={1} />
            <text x={W - padR} y={sy(r.y) - 4} fontSize="10" fill={INK} textAnchor="end">{r.label}</text>
          </g>
        ))}
        {series.map((s, si) => {
          const pts = s.points.map((p) => `${sx(p.x)},${sy(Math.min(p.y, ym))}`).join(" ");
          const last = s.points[s.points.length - 1];
          return (
            <g key={s.name}>
              <polyline points={pts} fill="none" stroke={SERIES[si]} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
              {last && <circle cx={sx(last.x)} cy={sy(Math.min(last.y, ym))} r={4} fill={SERIES[si]} stroke="#131720" strokeWidth={2} />}
            </g>
          );
        })}
        <text x={padL} y={H - 6} fontSize="10" fill={AXIS}>{xFormat(x0)}</text>
        <text x={W - padR} y={H - 6} fontSize="10" fill={AXIS} textAnchor="end">{xFormat(x1)}</text>
        {hover && base[hover.i] && (
          <line x1={sx(base[hover.i].x)} x2={sx(base[hover.i].x)} y1={padT} y2={H - padB} stroke={AXIS} strokeWidth={1} />
        )}
      </svg>
      {series.length > 1 && (
        <div className="viz-legend">
          {series.map((s, i) => (
            <span key={s.name}><i style={{ background: SERIES[i] }} />{s.name}</span>
          ))}
        </div>
      )}
      {hover && base[hover.i] && (
        <Tip x={hover.px} y={hover.py} width={ref.current?.clientWidth ?? 560}>
          <div className="muted">{xFormat(base[hover.i].x)}</div>
          {series.map((s, i) => (
            <div key={s.name}>
              {series.length > 1 && <i className="sw" style={{ background: SERIES[i] }} />}
              {s.name}: <b>{fmt(s.points[Math.min(hover.i, s.points.length - 1)]?.y ?? 0)}</b>
            </div>
          ))}
        </Tip>
      )}
    </div>
  );
}

/** Horizontal single-hue bars: identity is the text label, magnitude is the bar. */
export function BarList({ rows, unit = "", ariaLabel, color = SERIES[0] }: {
  rows: { label: string; value: number; note?: string }[];
  unit?: string;
  ariaLabel: string;
  color?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  if (!rows.length) return <div className="muted" style={{ fontSize: 12 }}>No data yet.</div>;
  const max = Math.max(...rows.map((r) => r.value), 1);
  const rowH = 26, labelW = 150, W = 560;
  const barW = W - labelW - 70;
  return (
    <svg viewBox={`0 0 ${W} ${rows.length * rowH + 4}`} width="100%" role="img" aria-label={ariaLabel}>
      {rows.map((r, i) => {
        const w = Math.max(r.value > 0 ? 4 : 0, (barW * r.value) / max);
        const y = i * rowH + 4;
        return (
          <g key={r.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <rect x={0} y={y - 2} width={W} height={rowH} fill={hover === i ? "#171d28" : "transparent"} />
            <text x={labelW - 8} y={y + 13} fontSize="11.5" fill={INK} textAnchor="end">{r.label}</text>
            <path d={`M${labelW},${y + 3} h${Math.max(0, w - 4)} a4,4 0 0 1 4,4 v6 a4,4 0 0 1 -4,4 h-${Math.max(0, w - 4)} z`}
              fill={color} />
            <text x={labelW + w + 6} y={y + 13} fontSize="11" fill={INK}>
              {fmt(r.value)}{unit}{hover === i && r.note ? ` · ${r.note}` : ""}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Vertical columns over an ordered category (histograms, per-task values). */
export function Columns({ rows, ariaLabel, unit = "", refLine, height = 150 }: {
  rows: { label: string; value: number | null; title?: string; status?: "ok" | "bad" | "warn" }[];
  ariaLabel: string;
  unit?: string;
  refLine?: { y: number; label: string };
  height?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  if (!rows.length) return <div className="muted" style={{ fontSize: 12 }}>No data yet.</div>;
  const W = 560, H = height, padL = 34, padB = 22, padT = 10;
  const max = niceMax(Math.max(...rows.map((r) => r.value ?? 0), refLine?.y ?? 0));
  const band = (W - padL - 8) / rows.length;
  const bw = Math.min(24, band - 2);
  const sy = (v: number) => padT + (1 - v / max) * (H - padT - padB);
  const statusFill = { ok: "#0ca30c", warn: "#fab219", bad: "#d03b3b" } as const;
  return (
    <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={ariaLabel}>
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line x1={padL} x2={W - 8} y1={sy(g * max)} y2={sy(g * max)} stroke={GRID} />
            <text x={padL - 5} y={sy(g * max) + 4} fontSize="10" fill={AXIS} textAnchor="end">{fmt(g * max)}</text>
          </g>
        ))}
        {refLine && (
          <g>
            <line x1={padL} x2={W - 8} y1={sy(refLine.y)} y2={sy(refLine.y)} stroke={AXIS} />
            <text x={W - 8} y={sy(refLine.y) - 4} fontSize="10" fill={INK} textAnchor="end">{refLine.label}</text>
          </g>
        )}
        {rows.map((r, i) => {
          const x = padL + i * band + (band - bw) / 2;
          const v = r.value ?? 0;
          const top = sy(v);
          const h = Math.max(v > 0 ? 4 : 0, H - padB - top);
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={padL + i * band} y={padT} width={band} height={H - padT - padB} fill="transparent" />
              {r.value === null ? (
                <text x={x + bw / 2} y={H - padB - 4} fontSize="9" fill={AXIS} textAnchor="middle">n/a</text>
              ) : (
                <path d={`M${x},${H - padB} v-${Math.max(0, h - 4)} a4,4 0 0 1 4,-4 h${bw - 8} a4,4 0 0 1 4,4 v${Math.max(0, h - 4)} z`}
                  fill={r.status ? statusFill[r.status] : SERIES[0]} opacity={hover === null || hover === i ? 1 : 0.55} />
              )}
              {rows.length <= 14 && (
                <text x={x + bw / 2} y={H - 8} fontSize="9.5" fill={AXIS} textAnchor="middle">{r.label}</text>
              )}
            </g>
          );
        })}
      </svg>
      {hover !== null && rows[hover] && (
        <div className="viz-tip" style={{ left: `${((padL + hover * band) / W) * 100}%`, top: 0 }}>
          {rows[hover].title || rows[hover].label}: <b>{rows[hover].value === null ? "n/a" : `${fmt(rows[hover].value!)}${unit}`}</b>
        </div>
      )}
    </div>
  );
}

/** Two measures of the same unit side by side per category (e.g. predicted vs actual LLM calls). */
export function PairedBars({ rows, names, ariaLabel }: {
  rows: { label: string; a: number; b: number }[];
  names: [string, string];
  ariaLabel: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const max = useMemo(() => niceMax(Math.max(1, ...rows.flatMap((r) => [r.a, r.b]))), [rows]);
  if (!rows.length) return <div className="muted" style={{ fontSize: 12 }}>No data yet.</div>;
  const W = 560, rowH = 30, labelW = 150, barW = W - labelW - 50;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${rows.length * rowH + 4}`} width="100%" role="img" aria-label={ariaLabel}>
        {rows.map((r, i) => {
          const y = i * rowH + 4;
          const wa = (barW * r.a) / max, wb = (barW * r.b) / max;
          return (
            <g key={i} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
              <rect x={0} y={y - 2} width={W} height={rowH} fill={hover === i ? "#171d28" : "transparent"} />
              <text x={labelW - 8} y={y + 15} fontSize="11" fill={INK} textAnchor="end">{r.label}</text>
              <rect x={labelW} y={y + 2} width={Math.max(r.a ? 3 : 0, wa)} height={10} rx={3} fill={SERIES[0]} />
              <rect x={labelW} y={y + 14} width={Math.max(r.b ? 3 : 0, wb)} height={10} rx={3} fill={SERIES[1]} />
              {hover === i && (
                <text x={labelW + Math.max(wa, wb) + 6} y={y + 17} fontSize="10.5" fill={INK}>
                  {names[0]} {fmt(r.a)} · {names[1]} {fmt(r.b)}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      <div className="viz-legend">
        <span><i style={{ background: SERIES[0] }} />{names[0]}</span>
        <span><i style={{ background: SERIES[1] }} />{names[1]}</span>
      </div>
    </div>
  );
}

export function StatTile({ label, value, note }: { label: string; value: ReactNode; note?: ReactNode }) {
  return (
    <div className="metric">
      <div className="k">{label}</div>
      <div className="v">{value}</div>
      {note && <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>{note}</div>}
    </div>
  );
}
