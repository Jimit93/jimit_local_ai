"""Governor state machine, admission control, caps; RAM reclaim safety."""
import subprocess
import sys
import threading
import time

import psutil
import pytest

from backend.agent import governor as G
from backend.agent import ram as RAM


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def gov(**kw):
    c = Clock()
    g = G.Governor(sampler=lambda: (0.0, 0.0, 8.0), interval_s=0.01, persist=False, clock=c, **kw)
    return g, c


def feed(g, c, cpu, n=1, own=0.0, ram=8.0):
    for _ in range(n):
        c.t += 1.0
        g.tick((cpu, own, ram))


def test_states_follow_thresholds():
    g, c = gov()
    feed(g, c, 20, 3)
    assert g.state == G.NORMAL
    feed(g, c, 65, 3)
    assert g.state == G.CAUTION
    feed(g, c, 75)  # one raw sample over the cap activates protection immediately
    assert g.state == G.HIGH_LOAD
    feed(g, c, 95)
    assert g.state == G.EMERGENCY


def test_recovery_needs_hysteresis_and_consecutive_samples():
    g, c = gov()
    feed(g, c, 80, 3)
    assert g.state == G.HIGH_LOAD
    feed(g, c, 68, 5)  # below cap but inside the hysteresis band
    assert g.state == G.HIGH_LOAD
    feed(g, c, 30, 2)
    assert g.state == G.HIGH_LOAD  # average still high / not enough consecutive samples
    feed(g, c, 30, 4)
    assert g.state in (G.NORMAL, G.CAUTION)
    ev = [e for e in g.protection_events if "recovered_at" in e]
    assert ev and ev[-1]["duration_s"] > 0


def test_low_ram_is_an_emergency(monkeypatch):
    monkeypatch.setattr(G.Governor, "_reclaim_agent_ram", lambda self: None)
    g, c = gov()
    feed(g, c, 10, 2, ram=0.3)
    assert g.state == G.EMERGENCY


def test_light_work_is_always_admitted_heavy_work_waits():
    g, c = gov()
    feed(g, c, 85, 3)
    assert g.admit("tool").admitted is True
    adm = g.admit("llm", timeout_s=0.2)
    assert adm.admitted is False and "HIGH_LOAD" in adm.reason
    assert any(e.get("state") == "DEFERRED" for e in g.protection_events)


def test_heavy_work_is_admitted_when_load_clears():
    g, c = gov()
    feed(g, c, 85, 3)
    result = {}

    def worker():
        result["adm"] = g.admit("tests", timeout_s=5)

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.2)
    feed(g, c, 20, 6)
    th.join(3)
    assert result["adm"].admitted is True and result["adm"].waited_ms > 100
    g.release("tests")


def test_concurrency_limit_and_slot_release():
    g, c = gov()
    feed(g, c, 65, 3)  # CAUTION: one heavy job at a time
    with g.slot("llm") as a:
        assert a.admitted
        assert g.admit("llm", timeout_s=0.1).admitted is False
    assert g.admit("llm", timeout_s=0.1).admitted is True


def test_critical_work_runs_under_high_load_but_not_emergency():
    g, c = gov()
    feed(g, c, 80, 3)
    assert g.admit("tests", critical=True, timeout_s=0.1).admitted is True
    g.release("tests")
    feed(g, c, 97, 2)
    assert g.admit("tests", critical=True, timeout_s=0.1).admitted is False


def test_child_cap_and_threads_leave_room_for_other_load():
    g, c = gov()
    feed(g, c, 30, 3, own=5)  # 25% used by other processes
    assert g.child_cpu_cap() == pytest.approx(40.0)  # 70 - 25 - 5
    assert 2 <= g.llm_threads() <= g.logical_cpus - 2
    feed(g, c, 85, 3, own=5)
    assert g.child_cpu_cap() <= 25.0


def test_policy_degrades_under_load():
    g, c = gov()
    feed(g, c, 10, 3)
    p_norm = g.policy()
    feed(g, c, 80, 3)
    p_high = g.policy()
    assert p_norm["allow_llm"] and not p_high["allow_llm"]
    assert p_high["context_scale"] < p_norm["context_scale"]
    assert p_high["prefer"] == "deterministic"


def test_real_sampler_produces_sane_values():
    s = G.psutil_sampler()
    time.sleep(0.2)
    cpu, own, ram = s()
    assert 0 <= cpu <= 100 and 0 <= own <= 100 and ram > 0


# ------------------------------------------------------------------ RAM

def test_protected_processes_are_never_candidates():
    rows = RAM.classify_processes(min_rss_mb=1)
    me = next(r for r in rows if r["pid"] == psutil.Process().pid)
    assert me["category"] == "protected"
    for r in rows:
        if (r["name"] or "").lower() in RAM.PROTECTED:
            assert r["category"] in ("protected", "agent_owned")


def test_allowlist_cannot_include_system_or_dev_tools(monkeypatch):
    monkeypatch.setenv("JLA_RAM_KILL_ALLOW", "lsass.exe;python.exe;SomeBigApp.exe")
    allow = RAM._allowlist()
    assert "lsass.exe" not in allow and "python.exe" not in allow and "somebigapp.exe" in allow


def test_reclaim_noop_when_enough_ram():
    r = RAM.reclaim(0.01, dry_run=True)
    assert r["ok"] and r["actions"] == []


def test_approved_kill_verifies_identity_and_terminates():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        info = psutil.Process(child.pid)
        wrong = RAM._terminate(child.pid, info.create_time() + 50, "python.exe")
        assert wrong["ok"] is False and "reused" in wrong["reason"]
        ok = RAM._terminate(child.pid, info.create_time(), info.name())
        assert ok["ok"] is True
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()


def test_reclaim_reports_shortfall_and_candidates_without_killing():
    huge = RAM.available_gb() + 1000
    r = RAM.reclaim(huge, dry_run=True, use_allowlist=False)
    assert r["ok"] is False and r["shortfall_gb"] > 900
    assert all(c["category"] == "candidate" for c in r["candidates"])
