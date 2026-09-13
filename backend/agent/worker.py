"""Background execution of agent tasks: queue, cancellation, approvals, resume.

One task runs at a time by default: CPU is the scarce resource on this machine, and
the governor already arbitrates heavy work inside a task. Every task still goes
through the harness entry point (`router.run_task`), so routing, metrics and
verification are identical to a synchronous request.
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque

from backend.agent import memory as MEM
from backend.agent import state as S
from backend.config import SETTINGS


class AgentWorker:
    def __init__(self, max_concurrent: int = 1, llm=None):
        self.max_concurrent = max_concurrent
        self.llm = llm                     # injected in tests; None = real code-model tiers
        self._q: deque[str] = deque()
        self._running: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.completed = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> "AgentWorker":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._dispatch, name="jla-agent-worker", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._lock:
            for info in self._running.values():
                info["cancel"].set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def submit(self, task_id: str) -> None:
        with self._lock:
            if task_id not in self._q and task_id not in self._running:
                self._q.append(task_id)
        self.start()
        self._wake.set()

    def status(self) -> dict:
        with self._lock:
            return {"queued": list(self._q), "running": [{"task_id": k, "started": v["started"]}
                                                          for k, v in self._running.items()],
                    "max_concurrent": self.max_concurrent, "completed": self.completed,
                    "alive": bool(self._thread and self._thread.is_alive())}

    def idle(self) -> bool:
        with self._lock:
            return not self._q and not self._running

    def wait_idle(self, timeout: float = 60.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if self.idle():
                return True
            time.sleep(0.05)
        return self.idle()

    # -------------------------------------------------------------- control
    def cancel(self, task_id: str) -> dict:
        with self._lock:
            if task_id in self._q:
                self._q.remove(task_id)
                S.set_status(task_id, S.CANCELLED, next_action="none")
                S.event(task_id, "stop", status=S.CANCELLED, reason="cancelled while queued")
                return {"cancelled": True, "was": "queued"}
            info = self._running.get(task_id)
        if info:
            info["cancel"].set()
            return {"cancelled": True, "was": "running", "note": "stops at the next step boundary"}
        st = S.load(task_id)
        if st["status"] in (S.AWAITING, S.INTERRUPTED, S.QUEUED):
            S.save(st, status=S.CANCELLED)
            S.event(task_id, "stop", status=S.CANCELLED, reason="cancelled by the user")
            return {"cancelled": True, "was": st["status"]}
        return {"cancelled": False, "status": st["status"]}

    def approve(self, task_id: str, approve: bool) -> dict:
        st = S.load(task_id)
        pending = st["data"].get("pending_approval")
        if st["status"] != S.AWAITING or not pending:
            return {"ok": False, "reason": f"task is {st['status']}, not awaiting approval"}
        S.event(task_id, "approval_decision", approved=approve, action=pending["action"], target=pending["target"])
        from backend.agent.policy import audit, RISK_HIGH

        audit(pending["action"], pending["target"], "approved" if approve else "deny", RISK_HIGH,
              "user decision", task_id=task_id)
        if not approve:
            st["data"].pop("pending_approval", None)
            st["last_output"] = f"{pending['action']} on {pending['target']} was denied by the user"
            S.save(st, status=S.CANCELLED)
            return {"ok": True, "status": S.CANCELLED}
        st["options"].setdefault("approved_actions", []).append(pending["action"])
        st["data"].pop("pending_approval", None)
        S.save(st, status=S.QUEUED)
        self.submit(task_id)
        return {"ok": True, "status": S.QUEUED}

    def resume(self, task_id: str) -> dict:
        analysis = S.analyze_resume(task_id)
        if not analysis.get("resumable"):
            return {"ok": False, **analysis}
        S.apply_resume(task_id, analysis)
        self.submit(task_id)
        return {"ok": True, **analysis}

    def recover(self, *, auto_resume: bool | None = None) -> list[dict]:
        """Startup: mark orphans interrupted; auto-resume only those that are safe."""
        auto = SETTINGS.agent.auto_resume if auto_resume is None else auto_resume
        out = []
        for a in S.recover_on_startup():
            a = dict(a)
            if auto and a.get("resumable") and a.get("safe_to_auto_resume"):
                try:
                    S.apply_resume(a["task_id"], a)
                    self.submit(a["task_id"])
                    a["auto_resumed"] = True
                except Exception as e:
                    a["auto_resumed"] = False
                    a["error"] = str(e)
            out.append(a)
        return out

    # ------------------------------------------------------------- dispatch
    def _dispatch(self) -> None:
        while not self._stop.is_set():
            task_id = None
            with self._lock:
                if self._q and len(self._running) < self.max_concurrent:
                    task_id = self._q.popleft()
                    cancel = threading.Event()
                    self._running[task_id] = {"cancel": cancel, "started": time.time()}
            if task_id is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            threading.Thread(target=self._run_one, args=(task_id, cancel), daemon=True,
                             name=f"jla-task-{task_id}").start()

    def _run_one(self, task_id: str, cancel: threading.Event) -> None:
        from backend.harness.router import run_task

        try:
            st = S.load(task_id)
            opts = {"workspace": st["workspace"], "task_id": task_id, "session_id": st.get("session_id"),
                    "allow_edits": bool(st["options"].get("allow_edits")),
                    "deep_review": bool(st["options"].get("deep_review")),
                    "cancel_event": cancel}
            if self.llm is not None:
                opts["llm"] = self.llm
            if st["plan"] and st["plan"][0]["kind"] == "harness":
                self._run_harness_task(st, opts)
            else:
                run_task(st["task"], options=opts)
                # A tracked task whose category escaped the coding chains is finalised here.
                st = S.load(task_id)
                if st["status"] in (S.QUEUED, S.RUNNING):
                    S.save(st, status=S.FAILED)
        except Exception as e:
            try:
                S.event(task_id, "error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
                S.set_status(task_id, S.FAILED, last_output=f"worker error: {type(e).__name__}: {e}")
            except Exception:
                pass
        finally:
            with self._lock:
                self._running.pop(task_id, None)
                self.completed += 1
            self._wake.set()

    def _run_harness_task(self, st: dict, opts: dict) -> None:
        """Non-coding requests (arithmetic, corpus questions, writing) typed into the agent chat."""
        from backend.harness.router import run_task

        tid = st["id"]
        step = st["plan"][0]
        step.update(status="running", started_at=time.time(), attempts=1)
        st["current_step"] = step["name"]
        S.save(st, status=S.RUNNING)
        S.event(tid, "step_start", step=step["name"], step_kind="harness")
        clean = {k: v for k, v in opts.items() if k not in ("task_id", "cancel_event", "llm")}
        out = run_task(st["task"], options=clean)
        ok = out["status"] in ("SUCCESS", "PARTIAL_SUCCESS", "LOW_CONFIDENCE") and not (out.get("error") and not out.get("answer"))
        layer = out.get("layer") or "deterministic"
        method = {"llm": "llm_general", "hybrid": "llm_general"}.get(layer, layer)
        step.update(status="done" if ok else "failed", ended_at=time.time(), method=method,
                    summary=(out.get("answer") or out.get("error") or "")[:500])
        c = st["counters"]
        c["llm_calls"] += out.get("llm_calls", 0)
        c["tokens"] += out.get("tokens", 0)
        c["iterations"] += 1
        if not out.get("llm_used"):
            c["deterministic_calls" if layer == "deterministic" else "nlp_calls"] = \
                c.get("deterministic_calls" if layer == "deterministic" else "nlp_calls", 0) + 1
        st["last_output"] = out.get("answer") or out.get("error") or ""
        st["validation_results"].append({"check": "harness", "ok": ok, "status": out["status"], "ts": time.time()})
        st["evaluation"] = {"predicted": {"llm_calls_max": 0 if out["task_type"] in ("deterministic", "structured_data", "retrieval", "text_processing", "ranking", "classification") else 1},
                            "actual": {"llm_calls": out.get("llm_calls", 0), "tokens": out.get("tokens", 0),
                                       "duration_s": round(out["latency_ms"] / 1000, 3), "processor": out.get("processor")},
                            "checks": {}, "match_score": None, "status": "completed" if ok else "failed"}
        pm = st["evaluation"]["predicted"]["llm_calls_max"]
        st["evaluation"]["checks"]["llm_budget_respected"] = out.get("llm_calls", 0) <= pm
        st["evaluation"]["match_score"] = float(st["evaluation"]["checks"]["llm_budget_respected"])
        S.event(tid, "step_end", step=step["name"], method=method, ok=ok, processor=out.get("processor"),
                task_type=out["task_type"], summary=(out.get("answer") or "")[:800], llm_calls=out.get("llm_calls", 0),
                tokens=out.get("tokens", 0), duration_ms=out["latency_ms"], trace=out.get("steps"),
                explanation=f"Harness category '{out['task_type']}' answered by {out.get('processor')}.")
        S.save(st, status=S.COMPLETED if ok else S.FAILED)
        S.event(tid, "complete", ok=ok)
        if st.get("session_id"):
            MEM.session_add(st["session_id"], "agent", st["last_output"][:1500], task_id=tid)


_worker: AgentWorker | None = None
_wlock = threading.Lock()


def worker() -> AgentWorker:
    global _worker
    with _wlock:
        if _worker is None:
            _worker = AgentWorker()
        return _worker


def set_worker(w: AgentWorker | None) -> None:
    global _worker
    with _wlock:
        if _worker is not None and _worker is not w:
            _worker.stop(1.0)
        _worker = w
