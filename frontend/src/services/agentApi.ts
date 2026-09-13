// Coding-agent API client. Same-origin /api (proxied to the local backend by Vite).
import { useEffect, useRef, useState } from "react";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api/agent${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) {
    let msg = await r.text();
    try {
      msg = JSON.parse(msg).detail ?? msg;
    } catch {
      /* plain text body */
    }
    throw new Error(`${r.status}: ${String(msg).slice(0, 300)}`);
  }
  return (await r.json()) as T;
}
const post = <T,>(p: string, body: unknown = {}) =>
  req<T>(p, { method: "POST", body: JSON.stringify(body) });
const q = (o: Record<string, string | number | undefined | null>) => {
  const s = Object.entries(o)
    .filter(([, v]) => v !== undefined && v !== null && v !== "")
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");
  return s ? `?${s}` : "";
};

export interface Step {
  name: string;
  kind: string;
  status: "pending" | "running" | "done" | "failed" | "skipped";
  attempts: number;
  method: string | null;
  reason: string | null;
  summary: string;
  checkpoint_id?: string | null;
}

export interface TaskState {
  id: string;
  workspace: string;
  task: string;
  topic: string;
  intent: string | null;
  session_id: string | null;
  status: string;
  current_step: string | null;
  plan: Step[];
  completed_steps: string[];
  pending_steps: string[];
  failed_steps: string[];
  last_checkpoint: string | null;
  active_files: string[];
  decisions: { ts: number; text: string }[];
  validation_results: Record<string, unknown>[];
  last_output: string;
  next_action: string;
  counters: Record<string, number>;
  options: Record<string, unknown>;
  data: Record<string, any>;
  evaluation?: {
    predicted: Record<string, any>;
    actual: Record<string, any>;
    checks: Record<string, any>;
    match_score: number | null;
  };
  resume_plan?: Record<string, any> | null;
  created_at: number;
  updated_at: number;
}

export interface AgentEvent {
  seq: number;
  ts: number;
  kind: string;
  step: string | null;
  method: string | null;
  payload: Record<string, any>;
}

export const agentApi = {
  status: () => req<any>("/status"),
  models: () => req<any>("/models"),
  workspaces: () => req<{ workspaces: string[] }>("/workspaces"),
  addWorkspace: (path: string) => post<any>("/workspaces", { path }),
  run: (body: { request: string; workspace?: string; session_id?: string; allow_edits?: boolean; deep_review?: boolean }) =>
    post<any>("/run", body),
  tasks: (params: { status?: string; session_id?: string; limit?: number } = {}) =>
    req<{ tasks: any[] }>(`/tasks${q(params)}`),
  task: (id: string) => req<TaskState>(`/tasks/${id}`),
  events: (id: string, after = 0) => req<{ events: AgentEvent[] }>(`/tasks/${id}/events${q({ after })}`),
  cancel: (id: string) => post<any>(`/tasks/${id}/cancel`),
  approve: (id: string, approve: boolean) => post<any>(`/tasks/${id}/approve`, { approve }),
  resumeAnalysis: (id: string) => req<any>(`/tasks/${id}/resume`),
  resume: (id: string) => post<any>(`/tasks/${id}/resume`),
  session: (id: string) => req<{ messages: any[] }>(`/sessions/${id}`),
  tree: (workspace?: string) => req<any>(`/files/tree${q({ workspace })}`),
  read: (path: string, workspace?: string) => req<any>(`/files/read${q({ path, workspace })}`),
  activity: (workspace?: string) => req<any>(`/files/activity${q({ workspace })}`),
  gitDiff: (workspace?: string, path?: string) => req<any>(`/git/diff${q({ workspace, path })}`),
  memory: (workspace?: string, search?: string) => req<any>(`/memory${q({ workspace, q: search })}`),
  addDecision: (text: string, workspace?: string) => post<any>("/memory/decision", { text, workspace }),
  checkpoints: (params: { task_id?: string; workspace?: string } = {}) => req<{ checkpoints: any[] }>(`/checkpoints${q(params)}`),
  checkpointDiff: (id: string) => req<any>(`/checkpoints/${id}/diff`),
  rollback: (id: string, force = false) => post<any>(`/checkpoints/${id}/rollback`, { force }),
  audit: (limit = 100) => req<{ audit: any[] }>(`/audit${q({ limit })}`),
  ram: () => req<any>("/ram"),
  reclaim: (required_gb: number, approved: any[] = []) => post<any>("/ram/reclaim", { required_gb, approved }),
  terminate: (pid: number, create_time: number, name: string) => post<any>("/ram/terminate", { pid, create_time, name }),
  setAllowlist: (names: string[]) => post<any>("/ram/allowlist", { names }),
  evaluation: () => req<any>("/evaluation"),
  index: (path: string) => post<any>("/index", { path }),
};

// ---------------------------------------------------------------- workspace choice

const WS_KEY = "jla.workspace";
export function getWorkspace(): string {
  try {
    return localStorage.getItem(WS_KEY) || "";
  } catch {
    return "";
  }
}
export function setWorkspace(ws: string) {
  try {
    localStorage.setItem(WS_KEY, ws);
  } catch {
    /* storage unavailable: selection lasts for this page only */
  }
  window.dispatchEvent(new CustomEvent("jla-workspace", { detail: ws }));
}
export function useWorkspace(): [string, (ws: string) => void] {
  const [ws, setWs] = useState(getWorkspace());
  useEffect(() => {
    const on = (e: Event) => setWs((e as CustomEvent).detail);
    window.addEventListener("jla-workspace", on);
    return () => window.removeEventListener("jla-workspace", on);
  }, []);
  return [ws, setWorkspace];
}

// ---------------------------------------------------------------------- polling

/** Poll while the tab is visible. Interval can change (fast while a task runs, slow when idle). */
export function usePoll<T>(fn: () => Promise<T>, intervalMs: number, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  useEffect(() => {
    let alive = true;
    let timer: number | undefined;
    const tick = async () => {
      // One-shot loads always run; only repeating polls pause while the tab is hidden.
      if (intervalMs <= 0 || document.visibilityState === "visible") {
        try {
          const d = await fnRef.current();
          if (alive) {
            setData(d);
            setError(null);
          }
        } catch (e) {
          if (alive) setError(String(e));
        }
      }
      if (alive && intervalMs > 0) timer = window.setTimeout(tick, intervalMs);
    };
    tick();
    return () => {
      alive = false;
      if (timer) window.clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, ...deps]);
  return { data, error, setData };
}
