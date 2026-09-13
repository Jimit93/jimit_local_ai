import { useMemo, useState } from "react";
import { Card, Err, Spinner } from "../components/Bits";
import { BarList, Columns, LineChart, PairedBars, StatTile } from "../components/AgentCharts";
import { METHOD_LABEL, TaskStatus } from "../components/AgentBits";
import { agentApi, usePoll } from "../services/agentApi";

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);

export default function AgentEval() {
  const ev = usePoll(() => agentApi.evaluation(), 15000);
  const [showTable, setShowTable] = useState(false);
  const d = ev.data?.tasks;
  const reports = ev.data?.reports || {};

  const recent = useMemo(() => [...(d?.tasks || [])].reverse().slice(-30), [d]);

  if (ev.error) return <Err>{ev.error}</Err>;
  if (!d) return <Spinner label="loading evaluation" />;

  const totals = d.totals || {};
  const cpu = (d.resources || []).map((r: any) => ({ x: r.ts, y: r.sys_cpu }));
  const ram = (d.resources || []).map((r: any) => ({ x: r.ts, y: r.ram_avail_gb }));
  const methodRows = Object.entries(d.methods || {})
    .map(([k, v]) => ({ label: METHOD_LABEL[k] || k, value: v as number }))
    .sort((a, b) => b.value - a.value);
  const callRows = [
    { label: "Tool calls", value: totals.tool_calls || 0 },
    { label: "NLP steps", value: totals.nlp_calls || 0 },
    { label: "Memory reuse", value: totals.ml_calls || 0 },
    { label: "Cache hits", value: totals.cache_hits || 0 },
    { label: "LLM calls", value: totals.llm_calls || 0 },
  ];

  return (
    <>
      <div className="page-head">
        <h2>Agent evaluation</h2>
        <p>Measured from recorded tasks, governor samples and backtest runs. Nothing on this page is estimated.</p>
      </div>

      <div className="metrics" style={{ marginBottom: 16 }}>
        <StatTile label="Tasks" value={d.n} note={Object.entries(d.statuses).map(([k, v]) => `${k} ${v}`).join(" · ")} />
        <StatTile label="Completion rate" value={pct(d.completion_rate)} />
        <StatTile label="Predicted vs actual" value={d.mean_match_score ?? "—"} note="mean match score, 0–1" />
        <StatTile label="Tasks with no LLM call" value={pct(d.llm_free_task_rate)} />
        <StatTile label="Tokens" value={(totals.tokens || 0).toLocaleString()} note={`${totals.llm_calls || 0} LLM calls`} />
        <StatTile label="Repairs" value={totals.repairs || 0} note={`${totals.iterations || 0} loop iterations`} />
      </div>

      <div className="grid-2">
        <Card title="Performance · system CPU %">
          <LineChart series={[{ name: "System CPU %", points: cpu }]} yMax={100} ariaLabel="System CPU percent over the last hour"
            refLines={[{ y: 70, label: "70% cap" }]} />
        </Card>
        <Card title="Performance · RAM available (GB)">
          <LineChart series={[{ name: "RAM available GB", points: ram }]} ariaLabel="Available RAM over the last hour"
            refLines={[{ y: 1, label: "1 GB floor" }]} />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Efficiency · where work was done">
          <BarList rows={callRows} ariaLabel="Tool, NLP, memory, cache and LLM call counts" />
          <div className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>Cheaper layers should dominate. LLM calls are the expensive row.</div>
        </Card>
        <Card title="Agent behaviour · method per step">
          <BarList rows={methodRows} ariaLabel="Processing method distribution across all steps" color="#199e70" />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Quality · predicted vs actual (match score per task)">
          <Columns ariaLabel="Match score per task, oldest to newest" refLine={{ y: 1, label: "perfect" }}
            rows={recent.map((t: any, i: number) => ({
              label: String(i + 1), value: t.match_score, title: `${t.topic} (${t.status})`,
              status: t.match_score === null ? undefined : t.match_score >= 0.99 ? "ok" : t.match_score >= 0.5 ? "warn" : "bad",
            }))} />
          <div className="muted" style={{ fontSize: 11.5 }}>Green = prediction fully matched · amber = partly · red = mostly missed. Hover for the task.</div>
        </Card>
        <Card title="Quality · LLM calls, budget vs used">
          <PairedBars names={["Budget (predicted max)", "Used"]} ariaLabel="Predicted LLM call budget versus actual calls per task"
            rows={recent.slice(-10).filter((t: any) => t.predicted).map((t: any) => ({
              label: String(t.topic).slice(0, 22), a: t.predicted?.llm_calls_max ?? 0, b: t.llm_calls,
            }))} />
        </Card>
      </div>

      <div className="grid-2">
        <Card title="Agent behaviour · duration per task (s)">
          <Columns ariaLabel="Task duration in seconds" unit=" s"
            rows={recent.map((t: any, i: number) => ({ label: String(i + 1), value: t.duration_s ?? 0, title: t.topic }))} />
        </Card>
        <Card title="Failure recovery · repairs per task">
          <Columns ariaLabel="Number of tasks by repair count"
            rows={Object.entries(d.repairs_histogram || {}).map(([k, v]) => ({ label: `${k} repairs`, value: v as number }))} />
        </Card>
      </div>

      <BacktestReports reports={reports} />

      <Card title="Task log" right={<button className="btn ghost chip" onClick={() => setShowTable(!showTable)}>
        {showTable ? "Hide" : "Show"} table</button>}>
        {showTable && (
          <div className="scroll-x">
            <table>
              <thead><tr><th>Task</th><th>Status</th><th className="num">s</th><th className="num">LLM</th><th className="num">Tools</th>
                <th className="num">Tokens</th><th className="num">Repairs</th><th className="num">Match</th><th className="num">CPU peak</th></tr></thead>
              <tbody>
                {(d.tasks || []).map((t: any) => (
                  <tr key={t.id}>
                    <td>{t.topic}<div className="muted mono">{t.intent}</div></td>
                    <td><TaskStatus status={t.status} /></td>
                    <td className="num">{t.duration_s}</td><td className="num">{t.llm_calls}</td>
                    <td className="num">{t.tool_calls}</td><td className="num">{t.tokens}</td>
                    <td className="num">{t.repairs}</td><td className="num">{t.match_score ?? "—"}</td>
                    <td className="num">{t.cpu_peak ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}

function BacktestReports({ reports }: { reports: Record<string, any> }) {
  const cpu = reports.cpu_backtest || reports.cpu_backtest_phase2;
  const rank = reports.ranking_eval;
  const sys = reports.system_backtest;
  if (!cpu && !rank && !sys) return null;
  const phases = cpu?.phases || {};
  const cpuRows = Object.entries(phases).filter(([, v]: any) => v && !Array.isArray(v) && v.avg !== undefined)
    .map(([k, v]: any) => ({ label: k.replace(/_/g, " "), value: v.avg, note: `peak ${v.peak}%` }));
  const threads = Array.isArray(phases.llm_threads) ? phases.llm_threads : [];
  return (
    <>
      {sys && (
        <Card title={`System backtest · ${new Date(sys.ts * 1000).toLocaleString()}`}>
          <BarList ariaLabel="System backtest pass rate by suite" unit="%"
            rows={Object.entries(sys.suites || {}).map(([k, v]: any) => ({ label: k, value: Math.round((v.pass_rate ?? 0) * 100),
              note: `${v.passed}/${v.total}` }))} />
        </Card>
      )}
      <div className="grid-2">
        {cpu && (
          <Card title="CPU backtest · average system CPU by phase (%)">
            <BarList rows={cpuRows} unit="%" ariaLabel="Average CPU by backtest phase" />
            {threads.length > 0 && (
              <table style={{ marginTop: 8 }}>
                <thead><tr><th className="num">Threads</th><th className="num">tok/s</th><th className="num">CPU avg</th><th className="num">CPU peak</th></tr></thead>
                <tbody>{threads.map((r: any) => (
                  <tr key={r.threads}><td className="num">{r.threads}</td><td className="num">{r.tokens_per_s}</td>
                    <td className="num">{r.avg}%</td><td className="num">{r.peak}%</td></tr>))}</tbody>
              </table>
            )}
            {cpu.power_plugged === false && <div className="muted" style={{ fontSize: 11.5, marginTop: 6 }}>Measured on battery power.</div>}
          </Card>
        )}
        {rank && (
          <Card title={`File ranking · ${rank.cases} generated cases`}>
            <PairedBars names={["BM25 only", "Fused ranker"]} ariaLabel="Ranking quality, BM25 versus fused"
              rows={[
                { label: "P@1", a: rank.bm25.p_at_1, b: rank.fused.p_at_1 },
                { label: "Recall@5", a: rank.bm25.recall_at_5, b: rank.fused.recall_at_5 },
                { label: "MRR", a: rank.bm25.mrr, b: rank.fused.mrr },
              ]} />
          </Card>
        )}
      </div>
    </>
  );
}
