"""CPU backtest: measure, do not assume, that the 70% policy holds.

Phases (each sampled with psutil at a fixed interval, system-wide):
  idle             nothing started by us
  job_cap          a deliberately CPU-hungry child under a Job Object hard cap
  llm_threads      llama.cpp generation at several thread counts (via the manager)
  protection       an uncapped foreign load pushes CPU past the cap; the governor must
                   enter HIGH_LOAD, defer heavy work, and recover after the load stops
  agent_workload   optional: callables supplied by the caller (normal / heavy tasks)

Results go to reports/agent/cpu_backtest.json. The burner used to create load is a
temporary script in %TEMP% and always runs inside a kill-on-close job.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Callable

import psutil

from backend.config import REPORTS_DIR, SETTINGS

OUT_DIR = REPORTS_DIR / "agent"

_BURNER = textwrap.dedent('''
    import multiprocessing as mp, sys, time
    def burn():
        x = 0
        while True:
            x += 1
    if __name__ == "__main__":
        n, dur = int(sys.argv[1]), float(sys.argv[2])
        ps = [mp.Process(target=burn, daemon=True) for _ in range(n)]
        for p in ps:
            p.start()
        time.sleep(dur)
        for p in ps:
            p.terminate()
''')


class CpuSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval = interval_s
        self.samples: list[tuple[float, float, float]] = []  # (ts, cpu, ram_avail_gb)
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self) -> "CpuSampler":
        psutil.cpu_percent(None)
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            cpu = psutil.cpu_percent(interval=self.interval)
            self.samples.append((time.time(), cpu, psutil.virtual_memory().available / 1e9))

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._t:
            self._t.join(timeout=self.interval * 3)

    def stats(self, skip_first: int = 0) -> dict:
        xs = [c for _, c, _ in self.samples[skip_first:]]
        if not xs:
            return {"n": 0}
        s = sorted(xs)
        return {"n": len(xs), "avg": round(statistics.fmean(xs), 1),
                "p50": round(s[len(s) // 2], 1), "p95": round(s[min(len(s) - 1, int(len(s) * 0.95))], 1),
                "peak": round(max(xs), 1), "min": round(min(xs), 1),
                "above_cap_fraction": round(sum(1 for x in xs if x > SETTINGS.agent.cpu_cap_percent) / len(xs), 3),
                "ram_min_gb": round(min(r for _, _, r in self.samples[skip_first:]), 2)}

    def series(self, t0: float) -> list[list[float]]:
        return [[round(ts - t0, 2), c] for ts, c, _ in self.samples]


def _burner_path() -> Path:
    p = Path(tempfile.gettempdir()) / "jla_cpu_burner.py"
    p.write_text(_BURNER, encoding="utf-8")
    return p


def measure_idle(duration_s: float = 6.0) -> dict:
    with CpuSampler() as s:
        time.sleep(duration_s)
    return {"phase": "idle", **s.stats()}


def measure_job_cap(cap_percent: float = 40.0, duration_s: float = 10.0) -> dict:
    """A child that wants ~100% of every core, hard-capped by a Job Object."""
    from backend.agent.proc import run_process

    n = psutil.cpu_count() or 4
    out: dict = {}
    t0 = time.time()
    with CpuSampler() as s:
        r = run_process([sys.executable, str(_burner_path()), str(n), str(duration_s)],
                        cwd=tempfile.gettempdir(), timeout_s=duration_s + 20,
                        cpu_rate_percent=cap_percent, label="cpu_backtest_burner")
    out.update(phase="job_cap", cap_percent=cap_percent, burner_processes=n,
               job_object=r.job_object, capped=r.cpu_cap_percent is not None,
               **s.stats(skip_first=3), series=s.series(t0))
    return out


def measure_uncapped_burst(duration_s: float = 4.0) -> dict:
    """Reference: the same burner without a cap (short), to show the cap is what holds."""
    from backend.agent.proc import run_process

    n = psutil.cpu_count() or 4
    with CpuSampler() as s:
        run_process([sys.executable, str(_burner_path()), str(n), str(duration_s)],
                    cwd=tempfile.gettempdir(), timeout_s=duration_s + 20, label="cpu_backtest_uncapped")
    return {"phase": "uncapped_reference", **s.stats(skip_first=2)}


def measure_llm_threads(thread_counts: list[int], *, max_tokens: int = 64,
                        prefer: str = "small") -> list[dict]:
    """Generation CPU and throughput per thread count. The model goes through the manager."""
    from backend.llm import manager as LLM

    prompt = "Explain in three sentences why deterministic tools should run before a language model."
    old_threads, old_pref = SETTINGS.llm.n_threads, SETTINGS.llm.prefer_size
    rows = []
    try:
        SETTINGS.llm.prefer_size = prefer
        for n in thread_counts:
            SETTINGS.llm.n_threads = n
            LLM.llm(force_reload=True)
            LLM.llm().load()
            LLM.generate("warm up", max_tokens=4, temperature=0.0)
            with CpuSampler(0.25) as s:
                r = LLM.generate(prompt, max_tokens=max_tokens, temperature=0.0)
            tps = r.completion_tokens / (r.latency_ms / 1000) if r.latency_ms else 0.0
            rows.append({"threads": n, "model": r.model, "latency_ms": round(r.latency_ms, 1),
                         "completion_tokens": r.completion_tokens, "tokens_per_s": round(tps, 2),
                         **s.stats(skip_first=1)})
            LLM.release()
    finally:
        SETTINGS.llm.n_threads, SETTINGS.llm.prefer_size = old_threads, old_pref
    return rows


def measure_protection(load_duration_s: float = 12.0, recover_window_s: float = 15.0,
                       cap_percent: float | None = None) -> dict:
    """Foreign load (uncapped, short) -> governor must protect, defer, then recover.

    `cap_percent` lowers the governor's thresholds for this run only. Needed when the
    machine cannot physically exceed the configured cap (e.g. on battery power a Core
    Ultra laptop tops out near 68% system CPU), otherwise activation is untestable.
    """
    from backend.agent.governor import HIGH_LOAD, EMERGENCY, Governor
    from backend.agent.proc import run_process

    g = Governor(interval_s=0.5, persist=False)
    if cap_percent is not None:
        g.cap, g.caution, g.emergency = cap_percent, cap_percent - 10, max(cap_percent + 20, 90)
    g.start()
    time.sleep(2.0)
    n = psutil.cpu_count() or 4
    events: dict = {"baseline_state": g.state}
    t_start = time.time()

    def load() -> None:
        run_process([sys.executable, str(_burner_path()), str(n), str(load_duration_s)],
                    cwd=tempfile.gettempdir(), timeout_s=load_duration_s + 20, label="foreign_load")
        events["load_stopped_at"] = time.time()

    th = threading.Thread(target=load, daemon=True)
    protected = (HIGH_LOAD, EMERGENCY)
    in_protection_s = 0.0
    with CpuSampler() as s:
        th.start()
        activated = None
        while th.is_alive():
            if g.state in protected:
                in_protection_s += 0.1
                if activated is None:
                    activated = time.time()
                    adm = g.admit("llm", timeout_s=2.0)
                    events["admission_during_load"] = adm.to_dict()
                    if adm.admitted:
                        g.release("llm")
                    events["child_cap_during_load"] = g.child_cpu_cap()
                    events["llm_threads_during_load"] = g.llm_threads()
            time.sleep(0.1)
        th.join(timeout=30)
        stop_at = events.get("load_stopped_at", time.time())
        recovered = None
        end = time.time() + recover_window_s
        while time.time() < end:
            if g.state not in protected:
                recovered = time.time()
                break
            time.sleep(0.1)
        if recovered:
            adm2 = g.admit("llm", timeout_s=2.0)
            events["admission_after_recovery"] = adm2.to_dict()
            if adm2.admitted:
                g.release("llm")
    g.stop()
    samples = list(g.samples)
    first_above = next((x.ts for x in samples if x.ts >= t_start and x.sys_cpu >= g.cap), None)
    first_act = next((t["ts"] for t in g.transitions if t["to"] in protected), None)
    last_exit = next((t["ts"] for t in reversed(g.transitions)
                      if t["from"] in protected and t["to"] not in protected and t["ts"] >= stop_at), None)
    load_s = max(0.1, stop_at - (first_above or t_start))
    return {
        "phase": "protection", "governor_cap_percent": g.cap, **s.stats(),
        "activated": first_act is not None,
        # From the first governor sample above the cap to the state change.
        "activation_latency_s": round(first_act - first_above, 2) if first_act and first_above else None,
        "load_start_to_activation_s": round(first_act - t_start, 2) if first_act else None,
        "fraction_of_load_in_protection": round(min(1.0, in_protection_s / load_s), 3),
        "recovered": recovered is not None,
        "recovery_time_s": round((last_exit or recovered) - stop_at, 2) if recovered else None,
        "transitions": list(g.transitions), "protection_events": list(g.protection_events),
        "series": s.series(t_start), **events,
    }


def measure_workload(name: str, fn: Callable[[], object]) -> dict:
    t0 = time.time()
    with CpuSampler() as s:
        err = None
        try:
            fn()
        except Exception as e:  # recorded, not raised: the backtest reports it
            err = f"{type(e).__name__}: {e}"
    return {"phase": name, "duration_s": round(time.time() - t0, 2), "error": err, **s.stats(),
            "series": s.series(t0)}


def write_report(report: dict, name: str = "cpu_backtest.json") -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p
