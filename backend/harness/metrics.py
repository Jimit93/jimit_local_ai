"""Observability: per-task records plus live process metrics.

Raw user content is never logged - only sizes, processors and costs.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import psutil

from backend.config import LOG_DIR, ensure_dirs
from backend.db import execute, query

_proc = psutil.Process()
_proc.cpu_percent(None)  # prime the counter so the first reading is meaningful


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def snapshot() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu_percent": _proc.cpu_percent(None),
        "system_cpu_percent": psutil.cpu_percent(None),
        "rss_mb": round(_proc.memory_info().rss / 1e6, 1),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "ram_used_percent": vm.percent,
    }


def record_task(
    *,
    task_id: str,
    task_type: str | None,
    processor: str | None,
    layer: str | None,
    confidence: float | None,
    latency_ms: float,
    llm_used: bool,
    llm_calls: int,
    tokens: int,
    escalations: int,
    success: bool,
    error_type: str | None = None,
    input_chars: int = 0,
) -> None:
    execute(
        "INSERT INTO task_runs(task_id,ts,task_type,processor,layer,confidence,latency_ms,"
        "llm_used,llm_calls,tokens,escalations,success,error_type,input_chars) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (task_id, time.time(), task_type, processor, layer, confidence, latency_ms,
         int(llm_used), llm_calls, tokens, escalations, int(success), error_type, input_chars),
    )
    ensure_dirs()
    Path(LOG_DIR / "tasks.jsonl").open("a", encoding="utf-8").write(
        json.dumps({
            "ts": round(time.time(), 3), "task_id": task_id, "task_type": task_type,
            "processor": processor, "layer": layer, "confidence": confidence,
            "latency_ms": round(latency_ms, 2), "llm_used": llm_used,
            "llm_calls": llm_calls, "tokens": tokens, "escalations": escalations,
            "success": success, "error_type": error_type, "input_chars": input_chars,
        }) + "\n"
    )


def summary(limit_days: float = 30.0) -> dict:
    since = time.time() - limit_days * 86400
    rows = query("SELECT * FROM task_runs WHERE ts >= ?", (since,))
    if not rows:
        return {
            "tasks": 0, "llm_calls": 0, "tokens": 0, "avg_latency_ms": 0.0,
            "llm_free_rate": 0.0, "success_rate": 0.0, "by_layer": {}, "by_processor": {},
        }
    n = len(rows)
    by_layer: dict[str, int] = {}
    by_proc: dict[str, int] = {}
    for r in rows:
        by_layer[r["layer"] or "?"] = by_layer.get(r["layer"] or "?", 0) + 1
        by_proc[r["processor"] or "?"] = by_proc.get(r["processor"] or "?", 0) + 1
    return {
        "tasks": n,
        "llm_calls": sum(r["llm_calls"] or 0 for r in rows),
        "tokens": sum(r["tokens"] or 0 for r in rows),
        "avg_latency_ms": round(sum(r["latency_ms"] or 0 for r in rows) / n, 1),
        "median_latency_ms": round(sorted(r["latency_ms"] or 0 for r in rows)[n // 2], 1),
        "llm_free_rate": round(sum(1 for r in rows if not r["llm_used"]) / n, 4),
        "success_rate": round(sum(1 for r in rows if r["success"]) / n, 4),
        "escalations": sum(r["escalations"] or 0 for r in rows),
        "by_layer": dict(sorted(by_layer.items(), key=lambda kv: -kv[1])),
        "by_processor": dict(sorted(by_proc.items(), key=lambda kv: -kv[1])[:12]),
    }


def recent(limit: int = 25) -> list[dict]:
    rows = query("SELECT * FROM task_runs ORDER BY ts DESC LIMIT ?", (limit,))
    return [dict(r) for r in rows]
