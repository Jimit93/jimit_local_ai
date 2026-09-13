"""Aggregate agent task records into the numbers the evaluation dashboard draws.

Everything here is computed from what the loop recorded (task state, step methods,
predicted-vs-actual evaluations, governor samples). Nothing is estimated.
"""
from __future__ import annotations

import json
import time
from collections import Counter

from backend.db import query


def aggregate_tasks(limit: int = 200, since_s: float | None = None) -> dict:
    sql = "SELECT id,status,intent,topic,created_at,updated_at,state_json FROM agent_tasks"
    params: list = []
    if since_s:
        sql += " WHERE created_at >= ?"
        params.append(time.time() - since_s)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    tasks, methods, statuses, intents = [], Counter(), Counter(), Counter()
    repairs_hist: Counter = Counter()
    totals = Counter()
    scores = []
    for r in query(sql, params):
        try:
            st = json.loads(r["state_json"])
        except (json.JSONDecodeError, TypeError):
            statuses["corrupted"] += 1
            continue
        c = st.get("counters", {})
        ev = st.get("evaluation") or {}
        step_methods = Counter(s.get("method") for s in st.get("plan", []) if s.get("method"))
        methods.update(step_methods)
        statuses[r["status"]] += 1
        intents[(r["intent"] or "?").split(":")[0] if (r["intent"] or "").startswith("harness") else (r["intent"] or "?")] += 1
        repairs_hist[c.get("repairs", 0)] += 1
        for k in ("llm_calls", "tokens", "tool_calls", "nlp_calls", "ml_calls", "deterministic_calls", "cache_hits",
                  "repairs", "iterations"):
            totals[k] += c.get(k, 0) or 0
        if ev.get("match_score") is not None:
            scores.append(ev["match_score"])
        act = ev.get("actual") or {}
        tasks.append({
            "id": r["id"], "topic": r["topic"], "intent": r["intent"], "status": r["status"],
            "created_at": r["created_at"], "duration_s": act.get("duration_s") or round((r["updated_at"] or 0) - (r["created_at"] or 0), 2),
            "llm_calls": c.get("llm_calls", 0), "tokens": c.get("tokens", 0), "tool_calls": c.get("tool_calls", 0),
            "repairs": c.get("repairs", 0), "iterations": c.get("iterations", 0), "cache_hits": c.get("cache_hits", 0),
            "match_score": ev.get("match_score"), "checks": ev.get("checks"), "predicted": ev.get("predicted"),
            "actual": act, "cpu_avg": act.get("cpu_avg"), "cpu_peak": act.get("cpu_peak"),
            "methods": dict(step_methods), "context_tokens_est": act.get("context_tokens_est"),
        })
    n = len(tasks)
    done = statuses.get("completed", 0)
    return {
        "n": n, "statuses": dict(statuses), "intents": dict(intents.most_common(20)),
        "methods": dict(methods), "totals": dict(totals), "repairs_histogram": {str(k): v for k, v in sorted(repairs_hist.items())},
        "completion_rate": round(done / n, 3) if n else None,
        "mean_match_score": round(sum(scores) / len(scores), 3) if scores else None,
        "llm_free_task_rate": round(sum(1 for t in tasks if not t["llm_calls"]) / n, 3) if n else None,
        "tasks": tasks,
        "resources": resource_history(),
    }


def resource_history(window_s: float = 3600, max_points: int = 240) -> list[dict]:
    rows = query("SELECT ts,sys_cpu,proc_cpu,ram_avail_gb,state FROM governor_samples WHERE ts >= ? ORDER BY ts",
                 (time.time() - window_s,))
    pts = [dict(r) for r in rows]
    if len(pts) > max_points:
        step = len(pts) / max_points
        pts = [pts[int(i * step)] for i in range(max_points)]
    return pts
