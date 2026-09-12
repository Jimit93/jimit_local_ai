"""Benchmark runner: LLM-only baseline vs the routing harness.

Both modes run the same task text against the same model, in the same process, so
the comparison is apples to apples. Every number written to the report is measured
here - nothing is estimated or carried over from a previous run.
"""
from __future__ import annotations

import json
import platform
import time
import uuid
from pathlib import Path

import psutil

from backend.benchmarks.tasks import BenchTask, build_tasks, grade
from backend.config import REPORTS_DIR, SETTINGS, ensure_dirs
from backend.db import execute
from backend.harness.router import run_task
from backend.llm import manager as LLM

_proc = psutil.Process()


def _measure(fn) -> tuple[dict, float, float, float]:
    """Run fn, returning (result, wall_ms, cpu_ms_delta, rss_mb_peak)."""
    cpu0 = _proc.cpu_times()
    t0 = time.perf_counter()
    out = fn()
    wall = (time.perf_counter() - t0) * 1000
    cpu1 = _proc.cpu_times()
    cpu_ms = ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) * 1000
    rss = _proc.memory_info().rss / 1e6
    return out, wall, cpu_ms, rss


def _run_one(task: BenchTask, mode: str) -> dict:
    def _go():
        return run_task(task.prompt, mode=mode, options=task.options, record=False)

    try:
        res, wall, cpu_ms, rss = _measure(_go)
        err = res.get("error") if not res.get("answer") else None
    except Exception as e:  # a crash in one task must not kill the suite
        return {
            "task_id": task.id, "suite": task.suite, "mode": mode, "correct": False,
            "score": 0.0, "latency_ms": 0.0, "cpu_ms": 0.0, "rss_mb": 0.0,
            "llm_calls": 0, "tokens": 0, "processor": None, "answer": "",
            "error": f"{type(e).__name__}: {e}", "explanation": "exception",
        }

    answer = res.get("answer") or ""
    if task.grade == "style":
        v = (res.get("validation") or {})
        score = float(v.get("style_match", 0.0)) if "style_match" in v else 0.0
        if mode == "baseline":
            # Grade the baseline's raw text with the same validator for fairness.
            from backend.services.style_engine import style_constraints
            from backend.harness.validator import validate_post

            v = validate_post(answer, style_constraints())
            score = v["style_match"]
        correct, expl = score >= 0.85, f"style_match {score}"
    else:
        correct, score, expl = grade(task, answer)

    return {
        "task_id": task.id, "suite": task.suite, "mode": mode,
        "correct": bool(correct), "score": round(float(score), 4),
        "latency_ms": round(wall, 1), "cpu_ms": round(cpu_ms, 1), "rss_mb": round(rss, 1),
        "llm_calls": res.get("llm_calls", 0), "tokens": res.get("tokens", 0),
        "processor": res.get("processor"), "layer": res.get("layer"),
        "escalations": res.get("escalations", 0),
        "answer": answer[:1200], "error": err, "explanation": expl,
    }


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    lat = sorted(r["latency_ms"] for r in rows)
    return {
        "tasks": n,
        "correct": sum(1 for r in rows if r["correct"]),
        "accuracy": round(sum(1 for r in rows if r["correct"]) / n, 4),
        "mean_score": round(sum(r["score"] for r in rows) / n, 4),
        "llm_calls": sum(r["llm_calls"] for r in rows),
        "tokens": sum(r["tokens"] for r in rows),
        "total_latency_ms": round(sum(lat), 1),
        "mean_latency_ms": round(sum(lat) / n, 1),
        "median_latency_ms": round(lat[n // 2], 1),
        "total_cpu_ms": round(sum(r["cpu_ms"] for r in rows), 1),
        "peak_rss_mb": round(max(r["rss_mb"] for r in rows), 1),
        "llm_free_tasks": sum(1 for r in rows if r["llm_calls"] == 0),
        "errors": sum(1 for r in rows if r["error"]),
    }


def run_benchmark(
    suites: list[str] | None = None,
    include_baseline: bool = True,
    limit: int | None = None,
    persist: bool = True,
) -> dict:
    ensure_dirs()
    tasks = [t for t in build_tasks() if not suites or t.suite in suites]
    if limit:
        tasks = tasks[:limit]

    run_id = uuid.uuid4().hex[:10]
    started = time.time()
    LLM.reset_usage()

    # Warm both paths first so cold-start cost is reported separately, not smeared
    # across whichever mode happens to run first.
    warm_t0 = time.perf_counter()
    warm = LLM.warmup()
    warm_ms = round((time.perf_counter() - warm_t0) * 1000, 1)

    rows: list[dict] = []
    for t in tasks:
        rows.append(_run_one(t, "harness"))
        if include_baseline:
            rows.append(_run_one(t, "baseline"))

    harness = [r for r in rows if r["mode"] == "harness"]
    baseline = [r for r in rows if r["mode"] == "baseline"]
    agg_h, agg_b = _agg(harness), _agg(baseline)

    def _red(b, h):
        return round((b - h) / b, 4) if b else None

    comparison = {}
    if baseline:
        comparison = {
            "llm_call_reduction": _red(agg_b["llm_calls"], agg_h["llm_calls"]),
            "token_reduction": _red(agg_b["tokens"], agg_h["tokens"]),
            "latency_reduction": _red(agg_b["total_latency_ms"], agg_h["total_latency_ms"]),
            "cpu_reduction": _red(agg_b["total_cpu_ms"], agg_h["total_cpu_ms"]),
            "accuracy_delta": round(agg_h["accuracy"] - agg_b["accuracy"], 4),
            "mean_score_delta": round(agg_h["mean_score"] - agg_b["mean_score"], 4),
        }

    by_suite: dict[str, dict] = {}
    for s in sorted({t.suite for t in tasks}):
        h = [r for r in harness if r["suite"] == s]
        b = [r for r in baseline if r["suite"] == s]
        by_suite[s] = {"harness": _agg(h), "baseline": _agg(b) if b else None}

    mf = LLM.selected_model()
    report = {
        "run_id": run_id,
        "started_at": started,
        "duration_s": round(time.time() - started, 1),
        "environment": {
            "os": platform.platform(),
            "cpu_logical": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "gpu": False,
            "model": mf.name if mf else (LLM.llm().name if LLM.is_loaded() else "unknown"),
            "model_size_gb": round(mf.total_size_bytes / 1e9, 2) if mf else None,
            "runtime": LLM.llm().runtime,
            "n_ctx": SETTINGS.llm.n_ctx,
            "n_threads": SETTINGS.llm.n_threads,
            "max_style_examples": SETTINGS.harness.max_style_examples,
        },
        "warmup": {**warm, "wall_ms": warm_ms},
        "totals": {"harness": agg_h, "baseline": agg_b or None},
        "comparison": comparison,
        "by_suite": by_suite,
        "results": rows,
    }

    if persist:
        Path(REPORTS_DIR / "benchmark_results.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        Path(REPORTS_DIR / "benchmark_report.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
        for r in rows:
            execute(
                "INSERT INTO benchmark_runs(run_id,ts,suite,task_id,mode,correct,score,"
                "latency_ms,llm_calls,tokens,processor,cpu_percent,rss_mb,error) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, started, r["suite"], r["task_id"], r["mode"], int(r["correct"]),
                 r["score"], r["latency_ms"], r["llm_calls"], r["tokens"], r["processor"],
                 None, r["rss_mb"], r["error"]),
            )
    return report


def render_markdown(rep: dict) -> str:
    e, t, c = rep["environment"], rep["totals"], rep["comparison"]
    h, b = t["harness"], t["baseline"]
    L: list[str] = []
    A = L.append
    A("# Benchmark: LLM-only baseline vs routing harness\n")
    A(f"Run `{rep['run_id']}` · {time.strftime('%Y-%m-%d %H:%M', time.localtime(rep['started_at']))}"
      f" · {rep['duration_s']}s total\n")
    A("## Environment\n")
    A(f"- Model: `{e['model']}` ({e['model_size_gb']} GB) via {e['runtime']}, "
      f"n_ctx {e['n_ctx']}, {e['n_threads']} threads")
    A(f"- Machine: {e['cpu_logical']} logical cores, {e['ram_total_gb']} GB RAM, GPU: {e['gpu']}")
    A(f"- Warm-up (excluded from task timings): {rep['warmup'].get('wall_ms')} ms\n")
    A("## Totals\n")
    A("| Metric | Baseline (LLM-only) | Harness | Change |")
    A("|---|---|---|---|")
    if b:
        def pct(x):
            return "—" if x is None else f"{x * 100:.1f}%"
        A(f"| Tasks | {b['tasks']} | {h['tasks']} | |")
        A(f"| Correct | {b['correct']} | {h['correct']} | "
          f"{c['accuracy_delta'] * 100:+.1f} pts |")
        A(f"| Accuracy | {b['accuracy'] * 100:.1f}% | {h['accuracy'] * 100:.1f}% | "
          f"{c['accuracy_delta'] * 100:+.1f} pts |")
        A(f"| LLM calls | {b['llm_calls']} | {h['llm_calls']} | −{pct(c['llm_call_reduction'])} |")
        A(f"| Tokens | {b['tokens']} | {h['tokens']} | −{pct(c['token_reduction'])} |")
        A(f"| Total latency | {b['total_latency_ms'] / 1000:.1f}s | "
          f"{h['total_latency_ms'] / 1000:.1f}s | −{pct(c['latency_reduction'])} |")
        A(f"| Median latency | {b['median_latency_ms']:.0f} ms | "
          f"{h['median_latency_ms']:.0f} ms | |")
        A(f"| CPU time | {b['total_cpu_ms'] / 1000:.1f}s | {h['total_cpu_ms'] / 1000:.1f}s | "
          f"−{pct(c['cpu_reduction'])} |")
        A(f"| Peak RSS | {b['peak_rss_mb']:.0f} MB | {h['peak_rss_mb']:.0f} MB | |")
        A(f"| Tasks with no LLM call | {b['llm_free_tasks']} | {h['llm_free_tasks']} | |")
    A("\n## By suite\n")
    A("| Suite | Harness acc. | Baseline acc. | Harness LLM calls | Baseline LLM calls |"
      " Harness latency | Baseline latency |")
    A("|---|---|---|---|---|---|---|")
    for s, v in rep["by_suite"].items():
        hh, bb = v["harness"], v["baseline"]
        b_acc = f"{bb['accuracy'] * 100:.0f}%" if bb else "—"
        b_calls = str(bb["llm_calls"]) if bb else "—"
        b_lat = f"{bb['total_latency_ms'] / 1000:.1f}s" if bb else "—"
        A(f"| {s} | {hh['accuracy'] * 100:.0f}% | {b_acc} | {hh['llm_calls']} | {b_calls} | "
          f"{hh['total_latency_ms'] / 1000:.1f}s | {b_lat} |")
    A("\n## Per task\n")
    A("| Task | Suite | Mode | Correct | Score | Processor | LLM calls | Tokens | Latency |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in rep["results"]:
        A(f"| {r['task_id']} | {r['suite']} | {r['mode']} | "
          f"{'yes' if r['correct'] else 'no'} | {r['score']:.2f} | "
          f"`{(r['processor'] or '-')[:34]}` | {r['llm_calls']} | {r['tokens']} | "
          f"{r['latency_ms']:.0f} ms |")
    A("\n### Grading\n")
    A("- `exact`: the answer must contain the value computed independently (SQL/Python).")
    A("- `set`: every required token must appear.")
    A("- `soft`: keyword-coverage heuristic — reported as a score, not as proof of quality.")
    A("- `style`: the deterministic style validator scores the generated post against the "
      "measured profile; the same validator grades the baseline's output.")
    return "\n".join(L) + "\n"
