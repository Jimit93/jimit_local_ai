"""Stress backtest: size, duration, bursts, failures, CPU and RAM pressure.

    python -m scripts.stress_backtest [--files 3000] [--session 60] [--skip-llm]

Uses its own database. Every scenario records pass/fail against an explicit
expectation plus the measured numbers. Writes reports/agent/stress_backtest.json.
"""
from __future__ import annotations

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--files", type=int, default=3000)
ap.add_argument("--session", type=int, default=60)
ap.add_argument("--skip-llm", action="store_true")
ap.add_argument("--db", default=os.path.join("data", "backtest", "stress_backtest.db"))
args = ap.parse_args()
os.makedirs(os.path.dirname(args.db), exist_ok=True)
os.environ["JLA_DB_PATH"] = os.path.abspath(args.db)

import json  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import psutil  # noqa: E402

from backend.agent import policy  # noqa: E402
from backend.agent import state as S  # noqa: E402
from backend.benchmarks.agent_backtest import make_fixture  # noqa: E402
from backend.config import REPORTS_DIR  # noqa: E402

RSS = lambda: round(psutil.Process().memory_info().rss / 1e6, 1)  # noqa: E731


def pct(xs, q):
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(len(s) * q))], 2) if s else None


def wait(task_id: str, timeout: float) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        st = S.load(task_id)
        if st["status"] in S.TERMINAL or st["status"] == S.AWAITING:
            return st
        time.sleep(0.2)
    return S.load(task_id)


def big_repo(n: int) -> dict:
    from backend.agent import code_tools as C
    from backend.agent import project_index as PI
    from backend.agent.context import build_code_context

    root = Path(tempfile.mkdtemp(prefix="jla_big_")) / "repo"
    rnd = random.Random(1)
    words = ["order", "invoice", "stock", "price", "customer", "cart", "ship", "tax", "refund", "report", "cache", "token"]
    for i in range(n):
        a, b = rnd.sample(words, 2)
        d = root / f"pkg{i % 40}" / f"mod{i // 40}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{a}_{b}_{i}.py").write_text(
            f'"""Handles {a} and {b} records."""\n\n\ndef compute_{a}_{i}(x):\n    """Compute {a} for {b}."""\n'
            f"    return x * {i % 7 + 1}\n\n\nclass {a.title()}{b.title()}{i}:\n    def run(self, rows):\n"
            f"        return [compute_{a}_{i}(r) for r in rows]\n", encoding="utf-8")
    for i in range(n // 15):
        d = root / "web" / f"feature{i % 10}"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"View{i}.tsx").write_text(f"export function View{i}() {{\n  return null;\n}}\n", encoding="utf-8")
    ws = policy.register_workspace(root)
    rss0 = RSS()
    t = time.perf_counter()
    b1 = PI.build(ws)
    build_s = time.perf_counter() - t
    t = time.perf_counter()
    b2 = PI.build(ws, force=False)
    rebuild_s = time.perf_counter() - t
    for p in list(root.rglob("*.py"))[:10]:
        p.write_text(p.read_text(encoding="utf-8") + "\n# touched\n", encoding="utf-8")
        os.utime(p, (time.time() + 2, time.time() + 2))
    t = time.perf_counter()
    b3 = PI.build(ws)
    touched_s = time.perf_counter() - t
    lat = []
    for q in ["where is the refund tax computed", "customer cart view", "compute_invoice_17", "stock report class", "ship token cache"] * 10:
        t = time.perf_counter()
        PI.rank_files(ws, q)
        lat.append((time.perf_counter() - t) * 1000)
    t = time.perf_counter()
    sc = C.search_code(ws, "compute_refund")
    search_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    tr = C.tree(ws, max_entries=5000)
    tree_ms = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    _, acct = build_code_context(ws, "fix compute_invoice_17 returning the wrong value")
    ctx_ms = (time.perf_counter() - t) * 1000
    out = {"files": b1["files"], "build_s": round(build_s, 2), "rebuild_unchanged_s": round(rebuild_s, 3),
           "rebuild_changed": b3["changed"], "rebuild_10_touched_s": round(touched_s, 3),
           "rank_ms_p50": pct(lat, 0.5), "rank_ms_p95": pct(lat, 0.95), "search_code_ms": round(search_ms, 1),
           "search_matches": sc["count"], "tree_ms": round(tree_ms, 1), "tree_truncated": tr["truncated"],
           "context_ms": round(ctx_ms, 1), "context_tokens": acct["tokens_est"], "rss_growth_mb": round(RSS() - rss0, 1)}
    out["passed"] = (b2["changed"] == 0 and b3["changed"] == 10 and out["rank_ms_p95"] < 1500 and out["context_tokens"] < 2500
                     and out["rebuild_unchanged_s"] < build_s)
    return out


def long_session(n: int, ws: str) -> dict:
    from backend.agent.service import submit_request

    reqs = ["Calculate 12 × 34", "does config.json exist?", "where is the function apply_discount defined", "git status",
            "show lines 1-5 of pricing.py", "where is get_stock used", "list the utils folder", "lint pricing.py"]
    lat, statuses, rss0 = [], {}, RSS()
    for i in range(n):
        t = time.perf_counter()
        sub = submit_request(reqs[i % len(reqs)], workspace=ws, session_id="stress_session")
        st = wait(sub["task_id"], 60)
        lat.append(time.perf_counter() - t)
        statuses[st["status"]] = statuses.get(st["status"], 0) + 1
    from backend.agent.memory import session_recent

    msgs = session_recent("stress_session", 1000)
    return {"requests": n, "statuses": statuses, "latency_s_p50": pct(lat, 0.5), "latency_s_p95": pct(lat, 0.95),
            "latency_s_max": round(max(lat), 2), "rss_growth_mb": round(RSS() - rss0, 1), "session_messages": len(msgs),
            "passed": statuses.get("completed", 0) == n and len(msgs) >= 2 * n}


def burst(ws: str, k: int = 15) -> dict:
    from backend.agent.service import submit_request
    from backend.agent.worker import worker

    max_running = 0
    stop = threading.Event()

    def watch():
        nonlocal max_running
        while not stop.is_set():
            max_running = max(max_running, len(worker().status()["running"]))
            time.sleep(0.02)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t = time.perf_counter()
    ids = [submit_request("where is get_stock used", workspace=ws, session_id="burst")["task_id"] for _ in range(k)]
    finals = [wait(i, 120)["status"] for i in ids]
    stop.set()
    return {"submitted": k, "completed": finals.count("completed"), "max_concurrent_running": max_running,
            "total_s": round(time.perf_counter() - t, 2), "passed": finals.count("completed") == k and max_running <= 1}


def tool_failures(ws: str) -> dict:
    from backend.agent.service import submit_request

    cases = {"missing file": "show me nothing_here.py", "invalid regex": "grep for [unclosed(",
             "escape": "show me ../../Windows/win.ini", "missing dir": "list the nowhere folder"}
    out = {}
    for name, req in cases.items():
        st = wait(submit_request(req, workspace=ws, session_id="failures")["task_id"], 60)
        out[name] = {"status": st["status"], "output": (st.get("last_output") or "")[:140], "llm_calls": st["counters"]["llm_calls"]}
    ok = all(v["status"] in ("failed", "completed", "blocked") and v["llm_calls"] == 0 for v in out.values())
    return {"cases": out, "passed": ok}


def cpu_pressure(ws: str) -> dict:
    """Heavy work under a real synthetic load, governor thresholds lowered to what this machine can reach."""
    from backend.agent.governor import Governor, set_governor
    from backend.agent.service import submit_request
    from backend.benchmarks.agent_cpu import _burner_path

    batt = psutil.sensors_battery()
    cap = 70.0 if (batt is None or batt.power_plugged) else 40.0
    g = Governor(interval_s=0.5, persist=False)
    g.cap, g.caution, g.emergency = cap, cap - 10, max(cap + 20, 90)
    set_governor(g)
    g.start()
    n = psutil.cpu_count() or 4
    burner = subprocess.Popen([sys.executable, str(_burner_path()), str(n), "25"], creationflags=0x00004000)
    time.sleep(4)
    t = time.perf_counter()
    st_id = submit_request("run the tests", workspace=ws, session_id="cpu_pressure")["task_id"]
    st = wait(st_id, 240)
    burner.wait(timeout=60)
    evs = S.events(st_id)
    deferred = [e for e in evs if e["kind"] == "governor"]
    out = {"cap_used": cap, "power_plugged": getattr(batt, "power_plugged", None), "status": st["status"],
           "duration_s": round(time.perf_counter() - t, 1), "governor_events": len(deferred),
           "protection_events": list(g.protection_events)[-5:], "transitions": [(x["from"], x["to"]) for x in g.transitions][-8:]}
    out["passed"] = st["status"] == "completed" and any(x["to"] in ("HIGH_LOAD", "EMERGENCY") for x in g.transitions)
    g.stop()
    set_governor(None)
    return out


def ram_pressure(ws: str) -> dict:
    from backend.agent import ram as RAM
    from backend.llm import manager as LLM

    capable = LLM.resolve_code_model("capable")
    need = round(capable["size_gb"] * 1.15 + 1.0, 2) if capable else None
    kills_before = sum(1 for a in policy.recent_audit(500) if a["action"] == "kill_process")
    report = RAM.reclaim(need, dry_run=True, use_allowlist=False) if need else {}
    try:
        r = LLM.generate_code("Reply with OK.", tier="capable", max_tokens=5, temperature=0)
        # Expected under pressure: the capable model does not fit and a smaller preferred one runs.
        outcome = {"ran": True, "model": r.model, "degraded_from": r.meta.get("degraded_from"), "ram": r.meta.get("ram")}
    except LLM.InsufficientRAM as e:
        outcome = {"ran": False, "insufficient_ram": True, "shortfall_gb": e.report.get("shortfall_gb"),
                   "candidates": len(e.report.get("candidates", []))}
    kills_after = sum(1 for a in policy.recent_audit(500) if a["action"] == "kill_process")
    return {"capable_model": capable, "needed_gb": need, "available_gb": RAM.available_gb(),
            "dry_run_shortfall_gb": report.get("shortfall_gb"), "outcome": outcome,
            "no_process_killed_without_approval": kills_after == kills_before,
            "passed": kills_after == kills_before and (outcome.get("ran") or outcome.get("insufficient_ram"))}


def model_failure(ws: str) -> dict:
    from backend.agent.service import submit_request
    from backend.llm import ollama_adapter

    old = ollama_adapter.BASE
    ollama_adapter.BASE = "http://127.0.0.1:9"  # nothing listens here
    try:
        st = wait(submit_request("explain how the apply_discount function works", workspace=ws, session_id="model_fail",
                                 options={"allow_edits": False})["task_id"], 600)
    finally:
        ollama_adapter.BASE = old
    methods = [s.get("method") for s in st["plan"]]
    return {"status": st["status"], "methods": methods, "output": (st.get("last_output") or "")[:200],
            "passed": st["status"] in ("completed", "failed") and ("llm_general" in methods or st["status"] == "failed")}


def main() -> int:
    from backend.agent.governor import governor
    from backend.agent.worker import AgentWorker, set_worker

    governor().start()
    set_worker(AgentWorker())
    ws = str(make_fixture(Path(tempfile.mkdtemp(prefix="jla_stress_")) / "shop"))
    policy.register_workspace(ws)
    rep: dict = {"ts": time.time(), "scenarios": {}}
    steps = [("big_repo", lambda: big_repo(args.files)), ("long_session", lambda: long_session(args.session, ws)),
             ("burst_submissions", lambda: burst(ws)), ("tool_failures", lambda: tool_failures(ws)),
             ("cpu_pressure", lambda: cpu_pressure(ws)), ("ram_pressure", lambda: ram_pressure(ws))]
    if not args.skip_llm:
        steps.append(("model_failure", lambda: model_failure(ws)))
    for name, fn in steps:
        t = time.time()
        try:
            rep["scenarios"][name] = fn()
        except Exception as e:
            import traceback

            rep["scenarios"][name] = {"passed": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        rep["scenarios"][name]["wall_s"] = round(time.time() - t, 1)
        print(name, json.dumps(rep["scenarios"][name], default=str)[:700], flush=True)
    rep["passed"] = all(s.get("passed") for s in rep["scenarios"].values())
    (REPORTS_DIR / "agent").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "agent" / "stress_backtest.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print("ALL PASSED" if rep["passed"] else "SOME SCENARIOS FAILED")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
