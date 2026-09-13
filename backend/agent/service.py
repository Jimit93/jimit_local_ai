"""Request intake for the agent UI/API: route, resolve follow-ups, create the task, queue it."""
from __future__ import annotations

import uuid

from backend.agent import loop as L
from backend.agent import memory as MEM
from backend.agent import policy
from backend.agent import state as S
from backend.agent.worker import worker


def submit_request(request: str, *, workspace: str | None = None, session_id: str | None = None,
                   options: dict | None = None) -> dict:
    from backend.harness.task_classifier import classify_task

    ws = str(policy.resolve_workspace(workspace))
    session_id = session_id or uuid.uuid4().hex[:10]
    opts = dict(options or {})
    followup = MEM.resolve_followup(session_id, request)
    if followup:
        if followup["action"] == "resume_last" and followup.get("task_id"):
            MEM.session_add(session_id, "user", request, task_id=followup["task_id"])
            res = worker().resume(followup["task_id"])
            return {"task_id": followup["task_id"], "session_id": session_id, "followup": followup, "resume": res}
        if followup["action"] == "repeat_last" and followup.get("previous_request"):
            request = followup["previous_request"]
        # rollback_last flows through the normal path: the "undo" intent uses the session.

    cls = classify_task(request)
    if cls.result in ("workspace", "coding"):
        st = L.create_task(ws, request, session_id=session_id, options=opts)
    else:
        st = S.new_task(ws, request, session_id=session_id, intent=f"harness:{cls.result}",
                        plan=[S.make_step("harness", "harness")], options=opts)
        st["data"]["intent"] = {"intent": f"harness:{cls.result}", "confidence": cls.confidence,
                                "stage": cls.meta.get("stage"), "processor": cls.processor, "params": {},
                                "complexity": {}, "deterministic": cls.result != "reasoning",
                                "alternatives": [], "latency_ms": round(cls.latency_ms, 3)}
        S.save(st)
        S.event(st["id"], "understand", method=cls.layer, intent=f"harness:{cls.result}",
                confidence=cls.confidence, classifier=cls.processor,
                explanation=f"Not a coding request: routed to the harness category '{cls.result}'.")
        S.event(st["id"], "plan", steps=[{"name": "harness", "kind": "harness"}],
                explanation="One step: the existing routing harness answers it.")
        MEM.session_add(session_id, "user", request, task_id=st["id"])
    worker().submit(st["id"])
    return {"task_id": st["id"], "session_id": session_id, "category": cls.result,
            "intent": st.get("intent"), "plan": [{"name": s["name"], "kind": s["kind"]} for s in st["plan"]],
            "followup": followup}


def task_view(task_id: str) -> dict:
    """State for the UI: everything except bulky internals (full context text, raw output)."""
    st = S.load(task_id)
    data = dict(st.get("data", {}))
    for k in ("context", "reproduce", "_last_cache_key", "review_diff"):
        if k in data and isinstance(data[k], str) and len(data[k]) > 2000:
            data[k] = data[k][:2000] + " …"
    if isinstance(data.get("reproduce"), dict):
        data["reproduce"] = {k: v for k, v in data["reproduce"].items() if k != "output"}
    prop = data.get("proposal")
    if isinstance(prop, dict):
        data["proposal"] = {k: v for k, v in prop.items() if k != "raw"}
    st["data"] = data
    return st
