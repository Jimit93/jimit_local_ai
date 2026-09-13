// All calls go to the local FastAPI backend on 127.0.0.1. Nothing leaves the machine.

const BASE = "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`${r.status}: ${body.slice(0, 300)}`);
  }
  return (await r.json()) as T;
}

const post = <T,>(p: string, body: unknown) =>
  req<T>(p, { method: "POST", body: JSON.stringify(body) });

export interface TaskStep {
  processor: string;
  layer?: string;
  confidence: number;
  accepted: boolean;
  latency_ms: number;
  error?: string | null;
}

export interface TaskResult {
  task_id: string;
  mode: string;
  task_type: string;
  classifier: { processor: string; confidence: number };
  answer: string;
  result: unknown;
  processor: string | null;
  layer: string | null;
  confidence: number;
  llm_used: boolean;
  llm_calls: number;
  tokens: number;
  latency_ms: number;
  escalations: number;
  steps: TaskStep[];
  status: string;
  validation: Record<string, unknown>;
  error: string | null;
  resources: { cpu_percent: number; rss_mb: number; rss_delta_mb: number };
}

export interface WriterResult {
  text: string;
  validation: {
    style_match: number;
    status: string;
    issues: string[];
    checks: Record<string, boolean>;
    stats: Record<string, number | string | string[]>;
  };
  plan: Record<string, unknown>;
  style_brief: string;
  examples: { doc_id: number; created_at: string; score: number; preview: string }[];
  metrics: {
    llm_calls: number;
    tokens: number;
    prompt_tokens: number;
    latency_ms: number;
    model: string;
    repaired: boolean;
    source_count: number;
    truncated?: boolean;
  };
  hashtags?: string[];
  refine?: RefineInfo;
}

export interface RefineStep {
  iteration: number;
  stage: "generate" | "refine";
  check: string | null;
  fixer: string;
  kind: "deterministic" | "llm";
  accepted: boolean;
  note: string;
  status: string;
  style_match: number;
  overall: number;
  issues: string[];
  llm_calls: number;
}

export interface RefineInfo {
  stages: string[];
  iterations: number;
  max_iterations: number;
  llm_fixes_used: number;
  max_llm_fixes: number;
  accepted_steps: number;
  stopped_because: string;
  initial: { status: string; style_match: number; overall: number; issues: string[] };
  final: { status: string; style_match: number; overall: number; issues: string[] };
  score_curve: number[];
  trace: RefineStep[];
}

export interface Hit {
  doc_id: number;
  score: number;
  lexical: number;
  semantic: number;
  doc_type: string;
  created_at: string | null;
  url: string | null;
  topic: string | null;
  chunk_index: number;
  n_chunks: number;
  text: string;
}

export const api = {
  health: () => req<any>("/health"),
  system: () => req<any>("/system"),
  task: (task: string, mode = "harness", options: Record<string, unknown> = {}) =>
    post<TaskResult>("/task", { task, mode, options }),
  generate: (body: Record<string, unknown>) => post<WriterResult>("/writer/generate", body),
  improve: (body: Record<string, unknown>) => post<WriterResult>("/writer/improve", body),
  refine: (body: Record<string, unknown>) => post<WriterResult>("/writer/refine", body),
  refineBacktest: (body: Record<string, unknown> = {}) => post<any>("/writer/refine-backtest", body),
  refineBacktestLatest: () => req<any>("/writer/refine-backtest/latest"),
  knowledge: () => req<any>("/knowledge/stats"),
  search: (body: Record<string, unknown>) =>
    post<{ hits: Hit[]; count: number; latency_ms: number; mode: string }>(
      "/knowledge/search",
      body,
    ),
  styleProfile: () => req<any>("/style/profile"),
  metrics: () => req<any>("/metrics/summary"),
  benchmarkLatest: () => req<any>("/benchmark/latest"),
  benchmarkRun: (body: Record<string, unknown>) => post<any>("/benchmark/run", body),
  ingest: () => post<any>("/ingest", {}),
  evaluationReport: () => req<any>("/evaluation/report"),
  evaluationScore: (body: Record<string, unknown>) => post<any>("/evaluation/score", body),
  sendFeedback: (body: Record<string, unknown>) => post<any>("/feedback", body),
  listFeedback: () => req<any>("/feedback"),
  getSettings: () => req<any>("/settings"),
  setSettings: (body: Record<string, unknown>) => post<any>("/settings", body),
};
