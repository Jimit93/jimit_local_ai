"""RAM reclamation before and during heavy work (model load, inference, tests).

Tiers, cheapest and safest first:

  0. Agent-owned, automatic: unload idle local models (llama.cpp singleton, idle Ollama
     models via keep_alive=0) and stop leftover agent child processes.
  1. User allowlist, automatic: processes whose executable name the user has listed as
     killable (`ram_kill_allowlist` preference / JLA_RAM_KILL_ALLOW). Largest first,
     graceful terminate, then kill.
  2. Anything else, approval only: listed as candidates; killed only when the user
     approves that exact pid (pid + name + create_time must still match).

Never touched: protected system/security processes, the foreground window's
process, this server and its ancestors, other users' processes.
Every kill decision is written to the audit log.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import psutil

from backend.agent import policy

PROTECTED = {
    "system", "system idle process", "registry", "smss.exe", "csrss.exe", "wininit.exe",
    "winlogon.exe", "services.exe", "lsass.exe", "lsaiso.exe", "svchost.exe", "dwm.exe",
    "explorer.exe", "fontdrvhost.exe", "memcompression", "memory compression",
    "msmpeng.exe", "nissrv.exe", "securityhealthservice.exe", "securityhealthsystray.exe",
    "sgrmbroker.exe", "smartscreen.exe", "audiodg.exe", "ctfmon.exe", "sihost.exe",
    "taskhostw.exe", "runtimebroker.exe", "searchhost.exe", "searchindexer.exe",
    "startmenuexperiencehost.exe", "shellexperiencehost.exe", "textinputhost.exe",
    "applicationframehost.exe", "conhost.exe", "spoolsv.exe", "wudfhost.exe", "dllhost.exe",
    "lockapp.exe", "userinit.exe", "wlanext.exe", "mpdefendercoreservice.exe",
    "ollama.exe", "ollama app.exe",  # models are unloaded through Ollama's API instead
}
# Developer tooling: may be killed only with a per-pid approval, never via the allowlist
# (a blanket "python.exe" rule would take out unrelated work and this app's frontend).
APPROVAL_ONLY = {
    "claude.exe", "code.exe", "cursor.exe", "windowsterminal.exe", "openconsole.exe",
    "powershell.exe", "pwsh.exe", "cmd.exe", "node.exe", "python.exe", "pythonw.exe",
    "git.exe", "uvicorn.exe",
}
OLLAMA = "http://127.0.0.1:11434"


def _allowlist() -> set[str]:
    names = {n.strip().lower() for n in os.environ.get("JLA_RAM_KILL_ALLOW", "").split(";") if n.strip()}
    try:
        from backend.db import recall

        for row in recall("preference", "ram_kill_allowlist"):
            names |= {str(n).lower() for n in json.loads(row["value"])}
    except Exception:
        pass
    return names - PROTECTED - APPROVAL_ONLY


def set_allowlist(names: list[str]) -> list[str]:
    from backend.db import remember

    clean = sorted({n.strip().lower() for n in names if n and n.strip()} - PROTECTED - APPROVAL_ONLY)
    remember("preference", "ram_kill_allowlist", clean, source="user")
    return clean


def available_gb() -> float:
    return round(psutil.virtual_memory().available / 1e9, 2)


def _ancestors() -> set[int]:
    pids = {os.getpid()}
    try:
        p = psutil.Process()
        pids |= {a.pid for a in p.parents()}
    except psutil.Error:
        pass
    return pids


def _foreground_pid() -> int | None:
    try:
        import win32gui
        import win32process

        return win32process.GetWindowThreadProcessId(win32gui.GetForegroundWindow())[1]
    except Exception:
        return None


def _me() -> str | None:
    try:
        return psutil.Process().username()
    except psutil.Error:
        return None


def classify_processes(min_rss_mb: float = 150.0) -> list[dict]:
    """Every same-user process above `min_rss_mb`, labelled with what may be done to it."""
    me, keep, fg = _me(), _ancestors(), _foreground_pid()
    allow = _allowlist()
    from backend.agent.proc import owned_processes

    owned = {p["pid"] for p in owned_processes()}
    out = []
    for p in psutil.process_iter(["pid", "name", "username", "memory_info", "create_time"]):
        info = p.info
        mi = info.get("memory_info")
        if not mi:
            continue
        rss_mb = mi.rss / 1e6
        if rss_mb < min_rss_mb:
            continue
        name = (info.get("name") or "").lower()
        if info["pid"] in owned:
            cat = "agent_owned"
        elif (name in PROTECTED or info["pid"] in keep or info["pid"] == fg
              or info.get("username") != me or info["pid"] <= 4):
            cat = "protected"
        elif name in allow:
            cat = "allowlisted"
        else:
            cat = "candidate"
        out.append({"pid": info["pid"], "name": info.get("name"), "rss_mb": round(rss_mb, 1),
                    "create_time": info.get("create_time"), "category": cat,
                    "foreground": info["pid"] == fg, "approval_only": name in APPROVAL_ONLY})
    return sorted(out, key=lambda r: -r["rss_mb"])


def _terminate(pid: int, create_time: float | None, expected_name: str | None, grace_s: float = 5.0) -> dict:
    try:
        p = psutil.Process(pid)
        if create_time is not None and abs(p.create_time() - create_time) > 1e-3:
            return {"pid": pid, "ok": False, "reason": "pid was reused by another process"}
        if expected_name and (p.name() or "").lower() != expected_name.lower():
            return {"pid": pid, "ok": False, "reason": "process name changed"}
        rss = p.memory_info().rss
        p.terminate()
        try:
            p.wait(timeout=grace_s)
            how = "terminated"
        except psutil.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)
            how = "killed"
        return {"pid": pid, "ok": True, "how": how, "freed_mb_est": round(rss / 1e6, 1)}
    except psutil.NoSuchProcess:
        return {"pid": pid, "ok": True, "how": "already_exited", "freed_mb_est": 0.0}
    except psutil.Error as e:
        return {"pid": pid, "ok": False, "reason": f"{type(e).__name__}: {e}"}


def ollama_loaded_models(timeout: float = 1.5) -> list[dict]:
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=timeout) as r:
            return json.loads(r.read()).get("models", [])
    except Exception:
        return []


def unload_ollama_model(name: str, timeout: float = 10.0) -> bool:
    body = json.dumps({"model": name, "keep_alive": 0}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True
    except Exception:
        return False


def reclaim_agent_owned(*, keep_models: set[str] | None = None, task_id: str | None = None) -> list[dict]:
    """Tier 0. Unload idle models and stop leftover agent processes."""
    actions = []
    keep = {m.lower() for m in (keep_models or set())}
    try:
        from backend.llm import manager as LLM

        if LLM.is_loaded() and not LLM.is_busy() and "llama.cpp" not in keep:
            name = LLM.release()
            actions.append({"tier": 0, "action": "unload_llamacpp", "model": name})
    except Exception as e:
        actions.append({"tier": 0, "action": "unload_llamacpp", "error": str(e)})
    for m in ollama_loaded_models():
        name = m.get("name") or m.get("model")
        if not name or name.lower() in keep:
            continue
        busy = False
        try:
            from backend.llm import manager as LLM

            busy = LLM.ollama_busy(name)
        except Exception:
            pass
        if busy:
            continue
        ok = unload_ollama_model(name)
        actions.append({"tier": 0, "action": "unload_ollama", "model": name, "ok": ok,
                        "size_gb": round((m.get("size") or 0) / 1e9, 2)})
    for a in actions:
        policy.audit("reclaim_ram", a.get("model"), "allow", policy.RISK_SAFE, json.dumps(a), task_id=task_id)
    return actions


def reclaim(required_gb: float, *, approved: list[dict] | None = None, use_allowlist: bool = True,
            keep_models: set[str] | None = None, dry_run: bool = False,
            task_id: str | None = None) -> dict:
    """Free RAM until `required_gb` is available. Returns what was done and what remains."""
    t0 = time.perf_counter()
    start = available_gb()
    report = {"required_gb": required_gb, "start_available_gb": start, "actions": [],
              "candidates": [], "dry_run": dry_run}
    if start >= required_gb:
        report.update(ok=True, end_available_gb=start, ms=0.0)
        return report

    if not dry_run:
        report["actions"] += reclaim_agent_owned(keep_models=keep_models, task_id=task_id)
        time.sleep(0.5)

    procs = classify_processes()
    if use_allowlist:
        for p in [p for p in procs if p["category"] == "allowlisted"]:
            if available_gb() >= required_gb:
                break
            if dry_run:
                report["actions"].append({"tier": 1, "action": "would_terminate", **p})
                continue
            r = _terminate(p["pid"], p["create_time"], p["name"])
            policy.audit("kill_process", f"{p['name']}({p['pid']})", "allow" if r["ok"] else "deny",
                         policy.RISK_HIGH, "user allowlist: " + json.dumps(r), task_id=task_id)
            report["actions"].append({"tier": 1, "action": "terminate", "name": p["name"], **r})

    for a in approved or []:
        match = next((p for p in procs if p["pid"] == a.get("pid")), None)
        if match is None or match["category"] == "protected":
            report["actions"].append({"tier": 2, "pid": a.get("pid"), "ok": False,
                                      "reason": "not found or protected"})
            continue
        if dry_run:
            report["actions"].append({"tier": 2, "action": "would_terminate", **match})
            continue
        r = _terminate(match["pid"], a.get("create_time", match["create_time"]), a.get("name", match["name"]))
        policy.audit("kill_process", f"{match['name']}({match['pid']})", "approved" if r["ok"] else "deny",
                     policy.RISK_HIGH, "user approved: " + json.dumps(r), task_id=task_id)
        report["actions"].append({"tier": 2, "action": "terminate", "name": match["name"], **r})

    end = available_gb()
    if end < required_gb:
        report["candidates"] = [p for p in classify_processes() if p["category"] == "candidate"][:15]
    report.update(ok=end >= required_gb, end_available_gb=end,
                  shortfall_gb=round(max(0.0, required_gb - end), 2),
                  ms=round((time.perf_counter() - t0) * 1000, 1))
    return report
