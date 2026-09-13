"""The controlled agent loop.

    OBSERVE -> UNDERSTAND -> PLAN -> SELECT METHOD -> EXECUTE -> VERIFY -> EVALUATE
            -> REPAIR (bounded) -> RE-VERIFY -> CHECKPOINT -> CONTINUE

Entry: a task row with a plan. Exit: every step done (completed), a step failed without
a repair path (failed), a limit hit or the same failure seen twice (blocked), a user
cancel (cancelled) or an approval needed (awaiting_approval). Limits: per-task iteration
budget, task deadline, per-step attempts, max repairs, governor admission for heavy work.

Method selection per step is cheapest-first: deterministic tool -> NLP -> ML/memory
-> small code model -> capable code model. Every choice is written to the timeline with
a one-line reason; no chain-of-thought is stored or shown.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from backend.agent import checkpoints as CP
from backend.agent import code_tools as C
from backend.agent import edits as E
from backend.agent import memory as MEM
from backend.agent import state as S
from backend.agent.intents import IntentResult, classify_intent
from backend.agent.planner import REPAIR_TARGET, plan_for
from backend.agent.policy import ApprovalRequired, PolicyError, redact
from backend.config import SETTINGS

LLMFn = Callable[..., object]

REASONS = {
    "deterministic": "the request maps to an exact tool; no language generation is needed",
    "nlp": "lexical ranking over the indexed code answers this without a model",
    "memory": "an earlier verified result for the same situation exists",
    "cache": "an identical request over identical code was already answered",
    "llm_small": "this needs code understanding or generation; the small code model is tried first",
    "llm_capable": "the task is complex or the small model's attempt failed verification",
    "llm_general": "no code model was reachable, so the local general model answered",
    "skipped": "not needed for this task",
}


class AwaitApproval(Exception):
    def __init__(self, action: str, target: str, reason: str):
        super().__init__(reason)
        self.action, self.target, self.reason = action, target, reason


class Deferred(Exception):
    pass


@dataclass
class Outcome:
    ok: bool
    summary: str = ""
    method: str = "deterministic"
    error: str | None = None
    data: dict = field(default_factory=dict)
    confidence: float = 1.0
    next: str | None = None          # None | "skip_rest" | "skip_to:<step>"
    skipped: bool = False
    llm_calls: int = 0
    tokens: int = 0


@dataclass
class Ctx:
    st: dict
    ir: IntentResult
    ws: str
    cancel: threading.Event | None
    llm: LLMFn
    step: dict | None = None

    @property
    def data(self) -> dict:
        return self.st["data"]


# ======================================================================= entry

def create_task(workspace: str, request: str, *, session_id: str | None = None, options: dict | None = None) -> dict:
    """UNDERSTAND + PLAN. Cheap, deterministic, persisted before anything runs."""
    opts = dict(options or {})
    ir = classify_intent(request)
    allow_edits = bool(opts.get("allow_edits", False))
    has_tests = C.detect_test_runner(workspace) is not None
    plan = plan_for(ir, allow_edits=allow_edits, has_tests=has_tests)
    st = S.new_task(workspace, request, session_id=session_id, intent=ir.intent, plan=plan, options=opts)
    st["data"]["intent"] = ir.to_dict()
    st["data"]["has_tests"] = has_tests
    st["predictions"] = []
    budget = len(plan) + SETTINGS.agent.max_repairs * 6 + 4
    st["limits"] = {"max_iterations": min(SETTINGS.agent.max_iterations, int(opts.get("max_iterations") or budget)),
                    "max_repairs": int(opts.get("max_repairs", SETTINGS.agent.max_repairs)),
                    "deadline": time.time() + float(opts.get("timeout_s") or SETTINGS.agent.task_timeout_s)}
    S.save(st)
    S.event(st["id"], "understand", method="deterministic" if ir.stage == "rules" else "ml",
            intent=ir.intent, confidence=ir.confidence, stage=ir.stage, classifier=ir.processor,
            params=ir.params, complexity=ir.complexity,
            explanation=f"Intent '{ir.intent}' from {'rules' if ir.stage == 'rules' else 'the TF-IDF/LogReg intent model'} "
                        f"({ir.confidence:.2f}); complexity {ir.complexity.get('level')}.")
    S.event(st["id"], "plan", steps=[{"name": s["name"], "kind": s["kind"]} for s in plan],
            explanation=f"{len(plan)} step(s); edits {'allowed' if allow_edits else 'not allowed (read-only)'}; "
                        f"tests {'detected' if has_tests else 'not detected'}.")
    if session_id:
        MEM.session_add(session_id, "user", request, task_id=st["id"])
    return st


def _default_llm(prompt: str, **kw):
    """Code-model tiers; if Ollama or every code model is unavailable, the general local model."""
    import urllib.error

    from backend.llm import manager as LLM

    try:
        return LLM.generate_code(prompt, **kw)
    except LLM.InsufficientRAM:
        raise
    except (RuntimeError, urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        r = LLM.generate(prompt, system=kw.get("system"), max_tokens=kw.get("max_tokens", 600),
                         temperature=kw.get("temperature", 0.1), finish=kw.get("finish", "prose"))
        r.meta.update({"tier": "general", "fallback_reason": f"{type(e).__name__}: {str(e)[:160]}"})
        return r


def run_agent(task_id: str, *, llm: LLMFn | None = None, cancel_event: threading.Event | None = None) -> dict:
    st = S.load(task_id)
    if st["status"] in S.TERMINAL:
        return st
    ir = IntentResult(**{k: v for k, v in st["data"]["intent"].items() if k in IntentResult.__dataclass_fields__})
    ctx = Ctx(st=st, ir=ir, ws=st["workspace"], cancel=cancel_event, llm=llm or _default_llm)
    st.setdefault("limits", {"max_iterations": SETTINGS.agent.max_iterations, "max_repairs": SETTINGS.agent.max_repairs,
                             "deadline": time.time() + SETTINGS.agent.task_timeout_s})
    if st["limits"]["deadline"] < time.time():  # resumed after a long pause: fresh deadline
        st["limits"]["deadline"] = time.time() + SETTINGS.agent.task_timeout_s
    st["data"].setdefault("t_start", time.time())
    S.save(st, status=S.RUNNING)
    try:
        return _loop(ctx)
    except Exception as e:  # last-resort guard: never leave a task "running" after a crash
        st["last_output"] = f"internal error: {type(e).__name__}: {e}"
        S.event(task_id, "error", error=st["last_output"])
        S.save(st, status=S.FAILED)
        return st


def _loop(ctx: Ctx) -> dict:
    st = ctx.st
    tid = st["id"]
    while True:
        S.heartbeat(tid)
        if ctx.cancel is not None and ctx.cancel.is_set():
            return _stop(ctx, S.CANCELLED, "cancelled by the user")
        if time.time() > st["limits"]["deadline"]:
            return _stop(ctx, S.BLOCKED, "task deadline reached")
        step = next((s for s in st["plan"] if s["status"] in ("pending", "running")), None)
        if step is None:
            return _finish(ctx)
        if st["counters"]["iterations"] >= st["limits"]["max_iterations"]:
            return _stop(ctx, S.BLOCKED, f"iteration limit ({st['limits']['max_iterations']}) reached")
        st["counters"]["iterations"] += 1
        step.update(status="running", started_at=time.time())
        step["attempts"] += 1
        st["current_step"] = step["name"]
        ctx.step = step
        S.save(st)
        S.event(tid, "step_start", step=step["name"], step_kind=step["kind"], attempt=step["attempts"])
        try:
            out = HANDLERS[step["kind"]](ctx)
        except AwaitApproval as a:
            step["status"] = "pending"
            step["attempts"] -= 1
            st["data"]["pending_approval"] = {"action": a.action, "target": a.target, "reason": a.reason,
                                              "step": step["name"], "ts": time.time()}
            st["next_action"] = f"waiting for approval: {a.action} {a.target}"
            S.event(tid, "approval", step=step["name"], action=a.action, target=a.target, reason=a.reason)
            return S.save(st, status=S.AWAITING)
        except Deferred as d:
            step["status"] = "pending"
            step["attempts"] -= 1
            st["counters"]["iterations"] -= 1
            S.event(tid, "governor", step=step["name"], reason=str(d))
            if time.time() > st["limits"]["deadline"]:
                return _stop(ctx, S.BLOCKED, f"deferred until deadline: {d}")
            time.sleep(2.0)
            continue
        except (PolicyError, ApprovalRequired) as e:
            out = Outcome(False, error=f"policy: {e}", method="deterministic")
        except Exception as e:
            out = Outcome(False, error=f"{type(e).__name__}: {e}", method="deterministic")

        _record(ctx, step, out)
        if out.ok:
            step["status"] = "skipped" if out.skipped else "done"
            if out.next == "skip_rest":
                for s in st["plan"]:
                    if s["status"] == "pending" and s["kind"] != "report":
                        s.update(status="skipped", summary="not needed")
            elif out.next and out.next.startswith("skip_to:"):
                target = out.next.split(":", 1)[1]
                for s in st["plan"]:
                    if s["name"] == target:
                        break
                    if s["status"] == "pending":
                        s.update(status="skipped", summary="not needed")
            S.save(st)
            continue

        # ---------------------------------------------------------- failure
        if step["attempts"] < step.get("max_attempts", 1):
            step["status"] = "pending"
            st["data"]["retry_feedback"] = out.error
            S.event(tid, "retry", step=step["name"], error=out.error, attempt=step["attempts"])
            S.save(st)
            continue
        target = REPAIR_TARGET.get(step["kind"])
        if target and any(s["name"] == target for s in st["plan"]):
            sig = MEM.error_signature(out.error or "")[:12]
            seen = st["error_signatures"].get(sig, 0)
            st["error_signatures"][sig] = seen + 1
            if seen >= 1:
                return _stop(ctx, S.BLOCKED, f"the same failure repeated after a repair ({(out.error or '')[:160]}); "
                                             "stopping instead of retrying the same approach")
            if st["counters"]["repairs"] >= st["limits"]["max_repairs"]:
                return _stop(ctx, S.FAILED, f"verification still failing after {st['counters']['repairs']} repair(s): "
                                            f"{(out.error or '')[:200]}")
            st["counters"]["repairs"] += 1
            st["data"]["last_failure"] = (out.error or "")[-3000:]
            st["data"]["tier_floor"] = "capable"  # escalate the model after a failed attempt
            reset = False
            for s in st["plan"]:
                if s["name"] == target:
                    reset = True
                if reset and s["kind"] != "report" and s["status"] in ("done", "failed", "running", "pending"):
                    s.update(status="pending", attempts=0)
            # Restore the files before re-proposing so the new edit applies to clean code.
            cp = next((s.get("checkpoint_id") for s in st["plan"] if s["kind"] == "apply_edit" and s.get("checkpoint_id")), None)
            if cp:
                rb = CP.rollback(cp, task_id=tid)
                S.event(tid, "rollback", step=step["name"], checkpoint=cp, ok=rb["ok"], conflicts=rb["conflicts"])
            S.event(tid, "repair", step=step["name"], back_to=target, repair=st["counters"]["repairs"],
                    error=(out.error or "")[:1500],
                    explanation=f"'{step['name']}' failed; returning to '{target}', the step responsible for the change.")
            S.save(st)
            continue
        step["status"] = "failed"
        if step.get("critical", True) is False:
            S.event(tid, "non_critical_failure", step=step["name"], error=out.error)
            step["status"] = "skipped"
            S.save(st)
            continue
        return _stop(ctx, S.FAILED, f"step '{step['name']}' failed: {out.error}")


def _record(ctx: Ctx, step: dict, out: Outcome) -> None:
    st = ctx.st
    step.update(ended_at=time.time(), method=out.method, summary=(out.summary or out.error or "")[:500],
                reason=REASONS.get(out.method, ""))
    step["actual"] = {"ok": out.ok, "llm_calls": out.llm_calls, "tokens": out.tokens,
                      "duration_ms": round((step["ended_at"] - (step["started_at"] or step["ended_at"])) * 1000, 1)}
    c = st["counters"]
    c["llm_calls"] += out.llm_calls
    c["tokens"] += out.tokens
    key = {"deterministic": "deterministic_calls", "nlp": "nlp_calls", "memory": "ml_calls", "cache": "cache_hits"}.get(out.method)
    if key:
        c[key] = c.get(key, 0) + 1
    if out.method in ("deterministic",) and step["kind"] not in ("report",):
        c["tool_calls"] += 1
    S.event(st["id"], "step_end", step=step["name"], method=out.method, ok=out.ok, skipped=out.skipped,
            summary=(out.summary or "")[:800], error=out.error, confidence=out.confidence,
            llm_calls=out.llm_calls, tokens=out.tokens, duration_ms=step["actual"]["duration_ms"],
            explanation=f"{METHOD_LABEL.get(out.method, out.method)} used. Reason: {REASONS.get(out.method, '')}.")


METHOD_LABEL = {"deterministic": "Deterministic tool", "nlp": "NLP ranking", "memory": "Verified memory",
                "cache": "Cached result", "llm_small": "Small code model", "llm_capable": "Capable code model",
                "llm_general": "General local model",
                "skipped": "Nothing"}


def _stop(ctx: Ctx, status: str, reason: str) -> dict:
    st = ctx.st
    st["status"] = status
    st["last_output"] = reason
    st["next_action"] = "none" if status == S.CANCELLED else f"blocked: {reason}"
    # A change that never passed verification must not stay in the user's files.
    if status in (S.FAILED, S.BLOCKED) and not st["options"].get("keep_failed_changes"):
        cps = [s.get("checkpoint_id") for s in st["plan"] if s["kind"] in ("apply_edit", "deterministic_fix")
               and s.get("checkpoint_id")]
        if cps and ctx.data.get("files_changed"):
            rb = CP.rollback(cps[0], task_id=st["id"])
            st["decisions"].append({"ts": time.time(), "ok": rb["ok"],
                                    "text": f"restored checkpoint {cps[0]} because the change did not pass verification"})
            S.event(st["id"], "rollback", checkpoint=cps[0], ok=rb["ok"], conflicts=rb["conflicts"],
                    explanation="Unverified change undone; the files are back to their state before the task.")
            if rb["ok"]:
                ctx.data["files_changed"] = []
                st["last_output"] = reason + f" — the unverified change was rolled back (checkpoint {cps[0]})."
    st["validation_results"].append({"ok": False, "reason": reason, "ts": time.time()})
    if status in (S.FAILED, S.BLOCKED):
        try:
            MEM.record_error(ctx.ws, reason, task_id=st["id"])
        except Exception:
            pass
    _evaluate(ctx)
    S.event(st["id"], "stop", status=status, reason=reason, checkpoint=st.get("last_checkpoint"))
    S.save(st, status=status)
    if st.get("session_id"):
        MEM.session_add(st["session_id"], "agent", f"[{status}] {reason}", task_id=st["id"])
    return st


def _finish(ctx: Ctx) -> dict:
    st = ctx.st
    S._derive(st)
    ok = not st["failed_steps"]
    st["status"] = S.COMPLETED if ok else S.FAILED
    _evaluate(ctx)
    S.event(st["id"], "complete", ok=ok, evaluation=st.get("evaluation"))
    S.save(st, status=S.COMPLETED if ok else S.FAILED)
    if st.get("session_id"):
        MEM.session_add(st["session_id"], "agent", st.get("last_output", "")[:1500], task_id=st["id"])
    return st


# =================================================================== helpers

def _governed(kind: str, ctx: Ctx, *, critical: bool = False, timeout_s: float = 90.0):
    from backend.agent.governor import governor

    adm = governor().admit(kind, critical=critical, timeout_s=timeout_s, cancel_event=ctx.cancel)
    if not adm.admitted:
        raise Deferred(adm.reason)
    if adm.waited_ms > 1:
        S.event(ctx.st["id"], "governor", step=ctx.step["name"] if ctx.step else None,
                reason=adm.reason, waited_ms=adm.waited_ms, state=adm.state)
    return adm


def _release(kind: str) -> None:
    from backend.agent.governor import governor

    governor().release(kind)


def _tiers(ctx: Ctx) -> list[str]:
    level = ctx.ir.complexity.get("level", "low")
    if ctx.data.get("tier_floor") == "capable" or level == "high":
        return ["capable", "small"]
    return ["small", "capable"]


def _llm_call(ctx: Ctx, prompt: str, *, system: str, max_tokens: int, temperature: float = 0.1,
              tier_order: list[str] | None = None, finish: str = "prose") -> tuple[object | None, str, str | None]:
    """Cache -> governor admission -> tiered model with RAM fallback. Returns (result, method, error)."""
    from backend.db import query, remember

    order = tier_order or _tiers(ctx)
    # The tier order is part of the key: an escalated retry must not be served the
    # cheaper model's cached answer.
    key = hashlib.sha1((",".join(order) + "\n" + system + "\n" + prompt).encode("utf-8", "ignore")).hexdigest()
    ctx.data["_last_cache_key"] = key
    hit = query("SELECT value FROM agent_memory WHERE kind='llm_cache' AND key=?", (key,))
    if hit:
        cached = json.loads(hit[0]["value"])
        return type("Cached", (), {"text": cached["text"], "model": cached["model"], "total_tokens": 0,
                                   "prompt_tokens": 0, "completion_tokens": 0, "meta": {"cached": True}})(), "cache", None
    errors = []
    for tier in order:
        adm = _governed("llm", ctx, timeout_s=120)
        try:
            scale = adm.degrade.get("max_tokens_scale", 1.0)
            r = ctx.llm(prompt, tier=tier, system=system, max_tokens=max(128, int(max_tokens * scale)),
                        temperature=temperature, task_id=ctx.st["id"], finish=finish)
        except Exception as e:
            name = type(e).__name__
            errors.append(f"{tier}: {name}: {str(e)[:200]}")
            if name == "InsufficientRAM":
                S.event(ctx.st["id"], "ram", step=ctx.step["name"], tier=tier, error=str(e),
                        candidates=getattr(e, "report", {}).get("candidates", [])[:8],
                        explanation="Not enough free RAM for this model; trying a smaller tier. Approve freeing "
                                    "RAM in the UI to allow the larger model.")
            continue
        finally:
            _release("llm")
        used = (r.meta or {}).get("tier") or tier
        method = {"capable": "llm_capable", "small": "llm_small", "general": "llm_general"}.get(used, "llm_small")
        remember("llm_cache", key, json.dumps({"text": r.text, "model": r.model}), source="llm", confidence=0.5)
        return r, method, None
    return None, "llm_small", "; ".join(errors) or "no model available"


def _is_ws_file(ws: str, rel: str) -> bool:
    try:
        return C.policy.resolve_path(ws, rel).is_file()
    except PolicyError:
        return False


def _is_test_path(rel: str) -> bool:
    return bool(re.search(r"(^|/)tests?/|(^|/)test_|_test\.py$|\.test\.|\.spec\.", rel))


def _error_text(ctx: Ctx) -> str:
    parts = []
    if ctx.ir.params.get("has_traceback"):
        parts.append(ctx.st["task"])
    rep = ctx.data.get("reproduce")
    if rep and not rep.get("ok"):
        parts.append("\n".join(f"{f['kind'].upper()} {f['nodeid']} - {f['message']}" for f in rep.get("failures", [])))
        parts.append(rep.get("output", "")[-2500:])
    if ctx.data.get("last_failure"):
        parts.append(ctx.data["last_failure"])
    return "\n".join(p for p in parts if p)


# ================================================================== handlers

def h_tool(ctx: Ctx) -> Outcome:
    from backend.agent.executor import run_intent_tool

    if ctx.ir.intent in ("rename_file", "delete_file", "rollback") and not ctx.st["options"].get("allow_edits"):
        return Outcome(False, error="read-only mode: enable edits to let the agent change files")
    approved = f"{ctx.ir.intent}" in ctx.st["options"].get("approved_actions", [])
    res = run_intent_tool(ctx.ir, ctx.st["task"], ctx.ws, task_id=ctx.st["id"], session_id=ctx.st.get("session_id"),
                          approved=approved, cancel_event=ctx.cancel)
    if res.error and res.error.startswith("approval_required"):
        raise AwaitApproval(ctx.ir.intent, str(res.meta.get("approval", {}).get("target")), res.error)
    if res.error and res.error.startswith("deferred"):
        raise Deferred(res.error)
    ctx.st["last_output"] = res.meta.get("answer") or res.error or ""
    if res.meta.get("files"):
        S.add_active_files(ctx.st, res.meta["files"])
    if isinstance(res.result, dict) and res.result.get("checkpoint_id"):
        ctx.st["last_checkpoint"] = res.result["checkpoint_id"]
    ctx.data["tool_result"] = {"processor": res.processor, "confidence": res.confidence}
    return Outcome(res.ok, summary=(res.meta.get("answer") or "")[:500], error=res.error,
                   confidence=res.confidence, method="deterministic")


def h_run_tests(ctx: Ctx) -> Outcome:
    _governed("tests", ctx, timeout_s=180)
    try:
        target = ctx.ir.params.get("test_target")
        r = C.run_tests(ctx.ws, target, task_id=ctx.st["id"], cancel_event=ctx.cancel)
    finally:
        _release("tests")
    ctx.data["reproduce"] = {k: r.get(k) for k in ("ok", "passed", "failed", "errors", "summary", "failures",
                                                   "locations", "duration_s", "exit_code")}
    ctx.data["reproduce"]["output"] = r.get("output", "")[-4000:]
    ctx.data["baseline_failures"] = sorted({f["nodeid"] for f in r.get("failures", [])})
    S.add_active_files(ctx.st, sorted({loc["path"] for loc in r.get("locations", [])})[:10])
    summ = r.get("summary") or f"exit {r['exit_code']}"
    note = "" if not r["ok"] or ctx.ir.intent != "fix_bug" else " (tests already pass; will rely on the request text)"
    return Outcome(True, summary=f"baseline tests: {summ}{note}", method="deterministic")


def h_locate(ctx: Ctx) -> Outcome:
    from backend.agent import project_index as PI

    ws, p = ctx.ws, ctx.ir.params
    PI.ensure_index(ws, max_age_s=5)
    if ctx.ir.intent == "rename_symbol":
        old = p.get("rename_from")
        if not old:
            return Outcome(False, error="could not read 'rename X to Y' from the request")
        hits = C.search_code(ws, rf"\b{re.escape(old)}\b", regex=True, glob="*.py", case_sensitive=True, limit=500)
        files = sorted({m["path"] for m in hits["matches"]})
        ctx.data["located"] = files
        S.add_active_files(ctx.st, files)
        return Outcome(bool(files), summary=f"'{old}' appears in {len(files)} Python file(s)", method="deterministic",
                       error=None if files else f"'{old}' not found")
    err = _error_text(ctx)
    ranking = PI.rank_files(ws, ctx.st["task"], top_k=8, error_text=err)
    ranked = [f["path"] for f in ranking["files"]]
    mentioned = [f for f in (p.get("files") or []) if _is_ws_file(ws, f)]
    tb_files = [loc["path"] for loc in PI.traceback_locations(err, ws)]
    tb_files += [loc["path"] for loc in (ctx.data.get("reproduce") or {}).get("locations", [])]
    tb_files = list(dict.fromkeys(f for f in tb_files if _is_ws_file(ws, f)))
    order = list(dict.fromkeys(mentioned + tb_files + ranked))
    src = [f for f in order if not _is_test_path(f)]
    tst = [f for f in order if _is_test_path(f)]
    # Which existing files a proposed edit may touch. A bug fix never edits the tests
    # (that is how a failing test gets "fixed" by weakening it) unless the request
    # names a test file; writing tests touches test files only.
    if ctx.ir.intent == "write_tests":
        editable = tst[:3]
    elif ctx.ir.intent == "fix_bug":
        editable = (src or [f for f in mentioned if _is_test_path(f)])[:3]
        # Naming the failing test is how people locate a bug; only an explicit statement
        # that the test itself is wrong makes a test file editable.
        if re.search(r"\b(?:the )?tests? (?:is|are|itself is) (?:wrong|incorrect|outdated|broken)\b|"
                     r"\bupdate the tests?\b|\bfix the test file\b", ctx.st["task"], re.I):
            editable += [f for f in mentioned if _is_test_path(f) and f not in editable]
    else:
        editable = (src + tst[:1])[:3]
    ctx.data["located"] = order[:8]
    ctx.data["allowed_files"] = editable
    ctx.data["ranking"] = {"confidence": ranking["confidence"], "files": ranking["files"][:6]}
    S.add_active_files(ctx.st, editable)
    S.refresh_fingerprint(ctx.st)
    method = "deterministic" if mentioned or tb_files else "nlp"
    return Outcome(bool(order) or ctx.ir.intent == "general_coding",
                   summary="relevant: " + ", ".join(order[:5]) + (f" · editable: {', '.join(editable)}" if editable else ""),
                   method=method, confidence=1.0 if method == "deterministic" else ranking["confidence"],
                   error=None if order or ctx.ir.intent == "general_coding" else "no relevant files found")


def h_context(ctx: Ctx) -> Outcome:
    from backend.agent.context import build_code_context
    from backend.agent.governor import governor

    scale = governor().policy()["context_scale"]
    items = MEM.memory_items_for(ctx.ws, ctx.st["task"])
    history = [d["text"] for d in ctx.st["decisions"][-5:] if isinstance(d, dict) and d.get("text")]
    text, acct = build_code_context(ctx.ws, ctx.st["task"], error_text=_error_text(ctx), memory_items=items,
                                    history=history, scale=scale, pinned_files=ctx.data.get("allowed_files", [])[:2])
    ctx.data["context"] = text
    ctx.data["context_acct"] = {k: acct[k] for k in ("context_chars", "tokens_est", "budget_chars", "files",
                                                      "blocks_included", "blocks_dropped", "redactions",
                                                      "whole_repo_chars_avoided")}
    if ctx.st["options"].get("allow_edits") and ctx.data.get("allowed_files"):
        py = [f for f in ctx.data["allowed_files"] if f.endswith(".py")]
        ctx.data["lint_before"] = sorted({f"{i['path']}:{i['message']}" for f in py for i in C.lint(ctx.ws, f)["issues"]})
    return Outcome(True, summary=f"context {acct['context_chars']} chars (~{acct['tokens_est']} tokens) from "
                                 f"{len(acct['files'])} file(s); {acct['whole_repo_chars_avoided']} chars of the repository not sent",
                   method="nlp")


_BACKTICK = re.compile(r"`([A-Za-z_][\w.]*(?:\(\))?)`")


def h_answer(ctx: Ctx) -> Outcome:
    from backend.harness.prompt_builder import SYSTEM_CODER, code_answer_prompt

    if ctx.ir.intent == "analyze_error":
        known = MEM.known_resolution(ctx.ws, ctx.st["task"])
        if known:
            ans = f"This error was seen before ({known.get('count')}x). It was resolved by: {known['resolution']}"
            ctx.st["last_output"] = ans
            return Outcome(True, summary=ans[:300], method="memory", confidence=0.92)
    prompt = code_answer_prompt(request=ctx.st["task"], context=ctx.data.get("context", ""), max_tokens=600)
    r, method, err = _llm_call(ctx, prompt, system=SYSTEM_CODER, max_tokens=600, temperature=0.2, finish="prose")
    if r is None:
        return Outcome(False, error=f"no model could answer: {err}", method="llm_small")
    text = redact(r.text.strip())[0]
    ctx.st["last_output"] = text
    ctx.data["answer_model"] = r.model
    return Outcome(bool(text), summary=text[:300], method=method, llm_calls=0 if method == "cache" else 1,
                   tokens=getattr(r, "total_tokens", 0), error=None if text else "empty answer")


def h_validate_answer(ctx: Ctx) -> Outcome:
    from backend.db import query
    from backend.harness.validator import validate_generic

    ans = ctx.st.get("last_output", "")
    v = validate_generic(ans)
    names = {n.rstrip("()").split(".")[-1] for n in _BACKTICK.findall(ans)}
    known_syms = {r["name"] for r in query("SELECT DISTINCT name FROM code_symbols WHERE workspace=?", (ctx.ws,))} if names else set()
    known_files = {r["path"].split("/")[-1] for r in query("SELECT path FROM code_files WHERE workspace=?", (ctx.ws,))} if names else set()
    unknown = sorted(n for n in names if n not in known_syms and n not in known_files and not re.fullmatch(r"[a-z]+", n))
    grounded = 1.0 if not names else round(1 - len(unknown) / len(names), 3)
    result = {"ok": v["status"] == "SUCCESS" and grounded >= 0.5, "generic": v, "referenced": sorted(names),
              "unknown_references": unknown, "grounded_ratio": grounded, "ts": time.time()}
    ctx.st["validation_results"].append(result)
    summary = f"answer checks: {v['status']}, {len(names)} code reference(s), {len(unknown)} not found in the index"
    return Outcome(True, summary=summary, method="deterministic", confidence=grounded)


def h_propose_edit(ctx: Ctx) -> Outcome:
    from backend.harness.prompt_builder import SYSTEM_CODER, code_edit_prompt

    allowed = ctx.data.get("allowed_files") or []
    allow_new = ctx.ir.intent in ("implement", "write_tests", "create_file", "refactor")
    err_text = _error_text(ctx)
    # Memory replay: the same failure was fixed before with edits that still apply cleanly.
    if err_text and not ctx.data.get("last_failure"):
        known = MEM.known_resolution(ctx.ws, err_text)
        if known and known.get("edits"):
            blocks = [E.EditBlock(**b) for b in known["edits"]]
            dry = E.apply_blocks(ctx.ws, blocks, allowed_files=None, allow_new_files=allow_new, dry_run=True)
            if dry.ok:
                ctx.data["proposal"] = {"blocks": [b.to_dict() for b in blocks], "model": "memory", "diff": dry.diff}
                return Outcome(True, summary="re-using a verified earlier fix for the same error", method="memory",
                               confidence=0.95)
    feedback = ctx.data.pop("retry_feedback", "") or ""
    failure = "\n".join(x for x in (ctx.data.get("last_failure", ""), feedback) if x)
    prompt = code_edit_prompt(request=ctx.st["task"], context=ctx.data.get("context", ""), failure=failure[-3000:],
                              allowed_files=allowed)
    order = _tiers(ctx) if ctx.step["attempts"] == 1 else ["capable", "small"]
    r, method, err = _llm_call(ctx, prompt, system=SYSTEM_CODER, max_tokens=900, temperature=0.1, tier_order=order,
                               finish="edits")
    if r is None:
        return Outcome(False, error=f"no model produced an edit: {err}", method="llm_small")
    llm_calls = 0 if method == "cache" else 1
    blocks = E.parse_blocks(r.text)
    dry = E.apply_blocks(ctx.ws, blocks, allowed_files=allowed or None, allow_new_files=allow_new, dry_run=True)
    ctx.data["proposal"] = {"blocks": [b.to_dict() for b in blocks], "model": r.model, "diff": dry.diff,
                            "raw": redact(r.text[:4000])[0]}
    if not dry.ok:
        # An inapplicable proposal must never be served from the cache again.
        from backend.db import execute

        execute("DELETE FROM agent_memory WHERE kind='llm_cache' AND key=?", (ctx.data.get("_last_cache_key"),))
        return Outcome(False, error="proposed edit is not applicable: " + "; ".join(dry.errors)[:1500], method=method,
                       llm_calls=llm_calls, tokens=getattr(r, "total_tokens", 0))
    return Outcome(True, summary=f"{len(blocks)} edit block(s) from {r.model} for {', '.join(dry.files_changed)}",
                   method=method, llm_calls=llm_calls, tokens=getattr(r, "total_tokens", 0))


def h_apply_edit(ctx: Ctx) -> Outcome:
    prop = ctx.data.get("proposal") or {}
    blocks = [E.EditBlock(**b) for b in prop.get("blocks", [])]
    allow_new = ctx.ir.intent in ("implement", "write_tests", "create_file", "refactor")
    paths = sorted({b.path for b in blocks})
    cp = CP.create(ctx.ws, f"before {ctx.step['name']} ({ctx.st['topic'][:60]})", task_id=ctx.st["id"],
                   step=ctx.step["name"], paths=[p for p in paths if C.policy.resolve_path(ctx.ws, p, for_write=True).exists()],
                   state={"iteration": ctx.st["counters"]["iterations"]})
    ctx.step["checkpoint_id"] = cp
    ctx.st["last_checkpoint"] = cp
    ctx.st["checkpoints"].append(cp)
    S.save(ctx.st)  # the checkpoint exists before any byte is written
    S.event(ctx.st["id"], "checkpoint", step=ctx.step["name"], checkpoint=cp, files=paths)
    rep = E.apply_blocks(ctx.ws, blocks, allowed_files=ctx.data.get("allowed_files") or None,
                         allow_new_files=allow_new, checkpoint_id=cp, task_id=ctx.st["id"])
    if not rep.ok:
        return Outcome(False, error="; ".join(rep.errors)[:1500], method="deterministic")
    ctx.step["after_hashes"] = S.fingerprint(ctx.ws, rep.files_changed)
    ctx.data["files_changed"] = sorted(set(ctx.data.get("files_changed", [])) | set(rep.files_changed))
    ctx.data["last_diff"] = rep.diff
    S.add_active_files(ctx.st, rep.files_changed)
    S.refresh_fingerprint(ctx.st)
    return Outcome(True, summary=f"applied to {', '.join(rep.files_changed)} (checkpoint {cp})", method="deterministic")


def h_deterministic_fix(ctx: Ctx) -> Outcome:
    p = ctx.ir.params
    files = ctx.data.get("located") or []
    cp = CP.create(ctx.ws, f"before rename {p.get('rename_from')} -> {p.get('rename_to')}", task_id=ctx.st["id"],
                   step=ctx.step["name"], paths=files)
    ctx.step["checkpoint_id"] = cp
    ctx.st["last_checkpoint"] = cp
    ctx.st["checkpoints"].append(cp)
    S.save(ctx.st)
    rep = E.rename_symbol(ctx.ws, p["rename_from"], p["rename_to"], files=files, checkpoint_id=cp, task_id=ctx.st["id"])
    if not rep.ok:
        return Outcome(False, error="; ".join(rep.errors), method="deterministic")
    ctx.step["after_hashes"] = S.fingerprint(ctx.ws, rep.files_changed)
    ctx.data["files_changed"] = rep.files_changed
    ctx.data["last_diff"] = rep.diff
    total = sum(a["occurrences"] for a in rep.applied)
    return Outcome(True, summary=f"renamed {total} occurrence(s) in {len(rep.files_changed)} file(s) without a model",
                   method="deterministic")


def h_verify_syntax(ctx: Ctx) -> Outcome:
    errors = []
    for rel in ctx.data.get("files_changed", []):
        r = C.syntax_check(ctx.ws, rel)
        errors += r["errors"]
    ok = not errors
    ctx.st["validation_results"].append({"check": "syntax", "ok": ok, "errors": errors[:5], "ts": time.time()})
    return Outcome(ok, summary="syntax OK" if ok else f"{len(errors)} syntax error(s)", method="deterministic",
                   error=None if ok else "; ".join(f"{e['path']}:{e.get('line')}: {e.get('error')}" for e in errors))


_SERIOUS = re.compile(r"undefined name|redefinition of unused|used before assignment|syntax error", re.I)


def h_verify_lint(ctx: Ctx) -> Outcome:
    before = set(ctx.data.get("lint_before", []))
    new = []
    for rel in [f for f in ctx.data.get("files_changed", []) if f.endswith(".py")]:
        for i in C.lint(ctx.ws, rel)["issues"]:
            key = f"{i['path']}:{i['message']}"
            if key not in before:
                new.append(i)
    serious = [i for i in new if _SERIOUS.search(i["message"])]
    ok = not serious
    ctx.st["validation_results"].append({"check": "lint", "ok": ok, "new_issues": [i["message"] for i in new][:10],
                                         "ts": time.time()})
    return Outcome(ok, summary=f"pyflakes: {len(new)} new issue(s), {len(serious)} serious", method="deterministic",
                   error=None if ok else "; ".join(f"{i['path']}:{i['line']}: {i['message']}" for i in serious))


def h_verify_tests(ctx: Ctx) -> Outcome:
    _governed("tests", ctx, critical=True, timeout_s=240)
    try:
        baseline = set(ctx.data.get("baseline_failures", []))
        target = ctx.ir.params.get("test_target")
        r = C.run_tests(ctx.ws, target, task_id=ctx.st["id"], cancel_event=ctx.cancel)
    finally:
        _release("tests")
    now = {f["nodeid"] for f in r.get("failures", [])}
    new_failures = sorted(now - baseline)
    still = sorted(now & baseline)
    if ctx.ir.intent == "fix_bug":
        ok = not now and not r.get("timed_out") and r.get("exit_code") in (0, 5)
    else:
        ok = not new_failures and not r.get("timed_out")
    ctx.data["tests_after"] = {k: r.get(k) for k in ("ok", "passed", "failed", "errors", "summary", "duration_s")}
    ctx.st["validation_results"].append({"check": "tests", "ok": ok, "summary": r.get("summary"),
                                         "new_failures": new_failures, "still_failing": still, "ts": time.time()})
    err = None
    if not ok:
        fails = "\n".join(f"{f['kind'].upper()} {f['nodeid']} - {f['message']}" for f in r.get("failures", [])[:10])
        err = f"tests failing after the change ({r.get('summary')}):\n{fails}\n{r.get('output', '')[-1500:]}"
    return Outcome(ok, summary=f"tests: {r.get('summary') or 'exit ' + str(r.get('exit_code'))}", method="deterministic",
                   error=err)


def h_diff(ctx: Ctx) -> Outcome:
    if not C.is_git_repo(ctx.ws):
        return Outcome(False, error="not a git repository; nothing to diff")
    st = C.git_status(ctx.ws)
    d = C.git_diff(ctx.ws)
    ctx.data["review_files"] = [f["path"] for f in st["files"]]
    ctx.data["review_diff"] = d["diff"]
    if not st["files"]:
        ctx.st["last_output"] = "No uncommitted changes to review."
        return Outcome(True, summary="no changes", method="deterministic", next="skip_rest")
    return Outcome(True, summary=f"{len(st['files'])} changed file(s), {d['files_changed']} with diffs", method="deterministic")


_DEBUG = re.compile(r"^\+(?!\+\+).*\b(?:breakpoint\(\)|pdb\.set_trace|console\.log\(|debugger;|print\()", re.M)
_TODO = re.compile(r"^\+(?!\+\+).*\b(?:TODO|FIXME|XXX|HACK)\b", re.M)


def h_static_review(ctx: Ctx) -> Outcome:
    diff = ctx.data.get("review_diff", "")
    findings = []
    for rel in [f for f in ctx.data.get("review_files", []) if f.endswith(".py")]:
        try:
            s = C.syntax_check(ctx.ws, rel)
            findings += [f"{e['path']}:{e.get('line')}: syntax error {e.get('error')}" for e in s["errors"]]
            findings += [f"{i['path']}:{i['line']}: {i['message']}" for i in C.lint(ctx.ws, rel)["issues"]
                         if _SERIOUS.search(i["message"]) or "imported but unused" in i["message"]]
        except Exception:
            continue
    added = "\n".join(ln for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    if redact(added)[1]:
        findings.append("possible secret added in the diff (redacted in this report)")
    findings += [f"debug statement added: {m.group(0)[1:].strip()[:100]}" for m in _DEBUG.finditer(diff)][:10]
    findings += [f"marker added: {m.group(0)[1:].strip()[:100]}" for m in _TODO.finditer(diff)][:10]
    src = [f for f in ctx.data.get("review_files", []) if f.endswith(".py") and "test" not in f]
    tests = [f for f in ctx.data.get("review_files", []) if "test" in f]
    if src and not tests:
        findings.append(f"{len(src)} source file(s) changed with no test changes")
    if added.count("\n") > 400:
        findings.append(f"large change: {added.count(chr(10))} added lines")
    ctx.data["static_findings"] = findings
    return Outcome(True, summary=f"{len(findings)} automated finding(s)", method="deterministic")


def h_llm_review(ctx: Ctx) -> Outcome:
    from backend.agent.governor import governor
    from backend.harness.prompt_builder import SYSTEM_CODER, code_review_prompt

    diff = ctx.data.get("review_diff", "")
    deep = ctx.st["options"].get("deep_review") or ctx.ir.complexity.get("level") in ("medium", "high")
    if not deep or len(diff) > 12000 or not governor().policy()["allow_llm"]:
        why = ("not requested (set deep review for a model pass)" if not deep else
               "diff too large for the local model's context" if len(diff) > 12000 else "CPU protection active")
        return Outcome(True, summary=f"model review skipped: {why}", method="skipped", skipped=True)
    prompt = code_review_prompt(diff=diff, findings="\n".join(ctx.data.get("static_findings", [])), max_tokens=500)
    r, method, err = _llm_call(ctx, prompt, system=SYSTEM_CODER, max_tokens=500, finish="prose")
    if r is None:
        return Outcome(False, error=err, method="llm_small")
    ctx.data["model_review"] = redact(r.text)[0]
    return Outcome(True, summary=r.text[:300], method=method, llm_calls=0 if method == "cache" else 1,
                   tokens=getattr(r, "total_tokens", 0))


def h_report(ctx: Ctx) -> Outcome:
    st, d = ctx.st, ctx.data
    intent = ctx.ir.intent
    if intent == "review":
        f = d.get("static_findings", [])
        st["last_output"] = (st.get("last_output") if not d.get("review_files") else
                             f"Reviewed {len(d.get('review_files', []))} changed file(s).\nAutomated findings:\n"
                             + ("\n".join(f"- {x}" for x in f) or "- none")
                             + (f"\n\nModel review:\n{d['model_review']}" if d.get("model_review") else ""))
    elif d.get("files_changed"):
        tests = d.get("tests_after") or {}
        st["last_output"] = (f"Changed {', '.join(d['files_changed'])}."
                             + (f" Tests: {tests.get('summary')}." if tests else "")
                             + f" Checkpoint {st.get('last_checkpoint')} restores the previous version.\n\n"
                             + (d.get("last_diff") or "")[:6000])
        err = _error_text(ctx)
        if err and (not tests or tests.get("ok")):
            blocks = (d.get("proposal") or {}).get("blocks")
            MEM.record_error(ctx.ws, err, resolution=f"edited {', '.join(d['files_changed'])}", task_id=st["id"])
            if blocks:
                from backend.agent.memory import _put, error_signature, ws_key

                sig = error_signature(err)
                _put("error_pattern", f"{ws_key(ctx.ws)}:{sig[:12]}",
                     {"signature": sig, "count": 1, "resolution": f"edited {', '.join(d['files_changed'])}",
                      "edits": blocks, "last_task": st["id"], "ts": time.time()}, source="derived")
        MEM.record_decision(ctx.ws, f"{intent}: {st['task'][:120]} -> changed {', '.join(d['files_changed'])}",
                            task_id=st["id"])
    return Outcome(True, summary=(st.get("last_output") or "")[:300], method="deterministic")


HANDLERS: dict[str, Callable[[Ctx], Outcome]] = {
    "tool": h_tool, "run_tests": h_run_tests, "locate": h_locate, "context": h_context, "answer": h_answer,
    "validate_answer": h_validate_answer, "propose_edit": h_propose_edit, "apply_edit": h_apply_edit,
    "deterministic_fix": h_deterministic_fix, "verify_syntax": h_verify_syntax, "verify_lint": h_verify_lint,
    "verify_tests": h_verify_tests, "diff": h_diff, "static_review": h_static_review, "llm_review": h_llm_review,
    "report": h_report,
}


# ================================================================ evaluation

def _cpu_window(t0: float, t1: float) -> dict:
    try:
        from backend.agent.governor import governor

        xs = [s for s in governor().samples if t0 <= s.ts <= t1]
        if xs:
            return {"cpu_avg": round(sum(s.sys_cpu for s in xs) / len(xs), 1), "cpu_peak": max(s.sys_cpu for s in xs),
                    "ram_min_gb": min(s.ram_avail_gb for s in xs), "samples": len(xs)}
    except Exception:
        pass
    return {"samples": 0}


def _evaluate(ctx: Ctx) -> None:
    """Predicted vs actual for the whole task, computed from what was recorded."""
    st, d, ir = ctx.st, ctx.data, ctx.ir
    modifying = any(s["kind"] in ("apply_edit", "deterministic_fix") for s in st["plan"]) or ir.intent in ("rename_file", "delete_file")
    predicted_files = (d.get("allowed_files") or [])[:1] if modifying and ir.intent != "rename_symbol" else (d.get("located") or []) if ir.intent == "rename_symbol" else []
    predicted = {
        "files_changed": sorted(predicted_files),
        "tests_pass": True if (modifying and d.get("has_tests")) else None,
        "llm_calls_max": 0 if ir.deterministic or ir.intent == "rename_symbol" else (1 + st["limits"]["max_repairs"] if modifying else 1),
        "method": "deterministic" if ir.deterministic or ir.intent == "rename_symbol" else "llm",
    }
    actual_files = sorted(d.get("files_changed", []))
    tests_after = d.get("tests_after")
    t1 = time.time()
    actual = {
        "files_changed": actual_files,
        "tests_pass": (bool(tests_after.get("ok")) if tests_after else None),
        "llm_calls": st["counters"]["llm_calls"], "tokens": st["counters"]["tokens"],
        "tool_calls": st["counters"]["tool_calls"], "repairs": st["counters"]["repairs"],
        "iterations": st["counters"]["iterations"], "cache_hits": st["counters"].get("cache_hits", 0),
        "duration_s": round(t1 - d.get("t_start", t1), 2),
        "context_tokens_est": (d.get("context_acct") or {}).get("tokens_est"),
        **_cpu_window(d.get("t_start", t1), t1),
    }
    pf, af = set(predicted["files_changed"]), set(actual_files)
    tp = len(pf & af)
    precision = tp / len(af) if af else (1.0 if not pf else 0.0)
    recall = tp / len(pf) if pf else (1.0 if not af else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # File names are only predictable when the change targets existing code; for new
    # files (implement/write tests/create) the prediction is "something changes".
    names_predictable = ir.intent in ("fix_bug", "refactor", "rename_symbol") or not modifying
    checks = {
        "files_f1": round(f1, 3) if names_predictable else float(bool(af) == modifying),
        "tests_match": None if predicted["tests_pass"] is None else predicted["tests_pass"] == actual["tests_pass"],
        "llm_budget_respected": actual["llm_calls"] <= predicted["llm_calls_max"],
        "no_llm_when_deterministic": (actual["llm_calls"] == 0) if predicted["method"] == "deterministic" else None,
    }
    scored = [v for v in (checks["files_f1"], checks["tests_match"], checks["llm_budget_respected"],
                          checks["no_llm_when_deterministic"]) if v is not None]
    score = round(sum(float(v) for v in scored) / len(scored), 3) if scored else None
    st["evaluation"] = {"predicted": predicted, "actual": actual, "checks": checks, "match_score": score,
                        "status": st["status"]}
