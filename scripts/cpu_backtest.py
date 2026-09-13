"""Run the CPU backtest. Usage: python -m scripts.cpu_backtest [--quick] [--no-llm]"""
from __future__ import annotations

import argparse
import json
import time

import psutil

from backend.benchmarks import agent_cpu as A
from backend.config import SETTINGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--threads", default="")
    ap.add_argument("--out", default="cpu_backtest.json")
    ap.add_argument("--protection-only", action="store_true")
    ap.add_argument("--protection-cap", type=float, default=None,
                    help="governor cap for the protection phase (use when the machine cannot exceed 70%%)")
    args = ap.parse_args()
    if args.protection_only:
        batt = psutil.sensors_battery()
        rep = {"ts": time.time(), "power_plugged": getattr(batt, "power_plugged", None),
               "phases": {"protection": A.measure_protection(cap_percent=args.protection_cap)}}
        p = rep["phases"]["protection"]
        print(json.dumps({k: p.get(k) for k in ("governor_cap_percent", "avg", "peak", "activated",
                                                "activation_latency_s", "load_start_to_activation_s",
                                                "fraction_of_load_in_protection", "recovered", "recovery_time_s",
                                                "admission_during_load", "admission_after_recovery",
                                                "child_cap_during_load")}, indent=1))
        print("wrote", A.write_report(rep, args.out))
        return

    n = psutil.cpu_count() or 4
    threads = [int(x) for x in args.threads.split(",") if x] or sorted({max(2, n - 2), int(n * 0.6), int(n * 0.45)}, reverse=True)
    batt = psutil.sensors_battery()
    rep: dict = {"ts": time.time(), "logical_cpus": n, "cap_percent": SETTINGS.agent.cpu_cap_percent,
                 "power_plugged": getattr(batt, "power_plugged", None), "phases": {}}
    print("idle…", flush=True)
    rep["phases"]["idle"] = A.measure_idle()
    print("uncapped reference…", flush=True)
    rep["phases"]["uncapped_reference"] = A.measure_uncapped_burst()
    print("job cap 40%…", flush=True)
    rep["phases"]["job_cap_40"] = A.measure_job_cap(40.0)
    print("job cap 60%…", flush=True)
    rep["phases"]["job_cap_60"] = A.measure_job_cap(60.0)
    print("governor protection…", flush=True)
    rep["phases"]["protection"] = A.measure_protection()
    if not args.no_llm:
        print(f"llm threads {threads}…", flush=True)
        rep["phases"]["llm_threads"] = A.measure_llm_threads(threads)
    path = A.write_report(rep, args.out)
    summary = {k: ({kk: v.get(kk) for kk in ("avg", "p95", "peak", "activated", "activation_latency_s",
                                             "recovered", "recovery_time_s", "capped")}
                   if isinstance(v, dict) else
                   [{kk: r.get(kk) for kk in ("threads", "avg", "p95", "peak", "tokens_per_s")} for r in v])
               for k, v in rep["phases"].items()}
    print(json.dumps(summary, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
