"""Memory layers, task state persistence, events, checkpoints-on-resume, corrupted state."""
import json
import time
from pathlib import Path

import pytest

from backend.agent import code_tools as C
from backend.agent import memory as M
from backend.agent import state as S
from backend.db import execute


# ------------------------------------------------------------------ memory

def test_session_memory_and_followups(workspace):
    M.session_add("s1", "user", "fix the add function", task_id=None)
    M.session_add("s1", "agent", "done", task_id="t1")
    M.session_add("s1", "user", "undo that")
    rec = M.session_recent("s1")
    assert [m["role"] for m in rec] == ["user", "agent", "user"]
    f = M.resolve_followup("s1", "undo that")
    assert f["action"] == "rollback_last" and f["task_id"] == "t1"
    assert M.resolve_followup("s1", "again")["previous_request"] == "fix the add function"
    assert M.resolve_followup("s1", "please add a docstring") is None


def test_project_memory_is_derived_and_scoped(workspace, tmp_path):
    facts = M.learn_project(workspace)
    assert facts["summary"]["test_runner"] == "pytest" and facts["test_command"] == "python -m pytest"
    pm = M.project_memory(workspace)
    assert pm["test_command"] == "python -m pytest"
    M.record_decision(workspace, "use exact string edits, never regex rewrites", task_id="t9")
    assert M.project_memory(workspace)["decisions"][0]["text"].startswith("use exact")


def test_error_patterns_and_known_resolution(workspace):
    err = 'File "C:\\x\\calc.py", line 6, in add\nAssertionError: assert -1 == 5'
    sig1 = M.record_error(workspace, err, resolution="changed a - b to a + b", task_id="t1")
    sig2 = M.error_signature('File "D:\\other\\calc.py", line 99, in add\nAssertionError: assert -7 == 12')
    assert sig1[:12] == sig2[:12]  # paths and numbers do not change the signature
    kr = M.known_resolution(workspace, err)
    assert kr and kr["resolution"].startswith("changed")


def test_memory_search_and_items(workspace):
    M.learn_project(workspace)
    M.record_decision(workspace, "the accumulator must stay immutable")
    hits = M.search("accumulator immutable", workspace=workspace)
    assert hits and hits[0]["kind"] == "decision"
    items = M.memory_items_for(workspace, "change the accumulator")
    assert any("Test command" in i for i in items) and any("immutable" in i for i in items)


def test_memory_redacts_secrets(workspace):
    M.record_decision(workspace, 'rotated db_password = "hunter2hunter2"')
    assert "hunter2hunter2" not in json.dumps(M.project_memory(workspace))


# ------------------------------------------------------------------- state

def _plan():
    return [S.make_step("inspect", "inspect"), S.make_step("apply_fix", "apply_edit"),
            S.make_step("verify", "run_tests")]


def test_state_roundtrip_and_derived_lists(workspace):
    st = S.new_task(workspace, "fix add", plan=_plan(), intent="fix_bug")
    st["plan"][0]["status"] = "done"
    st["plan"][1]["status"] = "failed"
    S.save(st, status=S.RUNNING)
    back = S.load(st["id"])
    assert back["completed_steps"] == ["inspect"] and back["failed_steps"] == ["apply_fix"]
    assert back["pending_steps"] == ["verify"] and back["status"] == S.RUNNING
    assert S.list_tasks(status=S.RUNNING)[0]["id"] == st["id"]


def test_events_are_ordered_and_redacted(workspace):
    st = S.new_task(workspace, "x", plan=_plan())
    S.event(st["id"], "step_start", step="inspect", method="deterministic")
    S.event(st["id"], "tool", step="inspect", output='token = "a1b2c3d4e5f6g7h8i9j0k1"')
    ev = S.events(st["id"])
    assert [e["seq"] for e in ev] == [1, 2]
    assert "a1b2c3d4e5f6g7h8i9j0k1" not in json.dumps(ev)
    assert S.events(st["id"], after_seq=1)[0]["kind"] == "tool"


def test_corrupted_state_is_detected_not_resumed(workspace):
    st = S.new_task(workspace, "x", plan=_plan())
    execute("UPDATE agent_tasks SET state_json=?, status='running' WHERE id=?", ("{not json", st["id"]))
    with pytest.raises(S.StateCorrupt):
        S.load(st["id"])
    a = S.analyze_resume(st["id"])
    assert a["resumable"] is False and "corrupted" in a["reason"]
    st2 = S.new_task(workspace, "y", plan=_plan())
    bad = dict(st2, plan="oops")
    execute("UPDATE agent_tasks SET state_json=? WHERE id=?", (json.dumps(bad), st2["id"]))
    assert S.analyze_resume(st2["id"])["resumable"] is False


def test_recover_on_startup_marks_orphans_interrupted(workspace):
    st = S.new_task(workspace, "fix add", plan=_plan())
    st["plan"][0]["status"] = "done"
    st["plan"][1]["status"] = "running"
    st["current_step"] = "apply_fix"
    S.save(st, status=S.RUNNING)
    execute("UPDATE agent_tasks SET heartbeat_at=? WHERE id=?", (time.time() - 120, st["id"]))
    out = S.recover_on_startup(stale_after_s=30)
    assert out and out[0]["task_id"] == st["id"]
    assert S.load(st["id"])["status"] == S.INTERRUPTED
    assert S.events(st["id"])[-1]["kind"] == "interrupted"


def test_resume_reinspects_when_project_changed_offline(workspace):
    st = S.new_task(workspace, "fix add", plan=_plan())
    S.add_active_files(st, ["calc.py"])
    S.refresh_fingerprint(st)
    st["plan"][0]["status"] = "done"
    st["plan"][1]["status"] = "pending"
    S.save(st, status=S.INTERRUPTED)
    Path(workspace, "calc.py").write_text("def add(a, b):\n    return a + b  # user fixed it\n", encoding="utf-8")
    a = S.analyze_resume(st["id"])
    assert a["resumable"] and a["project_changed"] and a["resume_step"] == "inspect"
    assert a["safe_to_auto_resume"] is False
    new = S.apply_resume(st["id"], a)
    assert [s["status"] for s in new["plan"]] == ["pending", "pending", "pending"] and new["status"] == S.QUEUED


def test_resume_after_edit_applied_skips_to_verification(workspace):
    st = S.new_task(workspace, "fix add", plan=_plan())
    S.add_active_files(st, ["calc.py"])
    r = C.edit_file(workspace, "calc.py", "return a - b", "return a + b", task_id=st["id"])
    st["plan"][0]["status"] = "done"
    st["plan"][1].update(status="running", checkpoint_id=r["checkpoint_id"],
                         after_hashes=S.fingerprint(workspace, ["calc.py"]))
    st["last_checkpoint"] = r["checkpoint_id"]
    S.refresh_fingerprint(st)
    S.save(st, status=S.INTERRUPTED)
    a = S.analyze_resume(st["id"])
    assert a["interrupted_step_applied"] is True and a["resume_step"] == "verify"
    new = S.apply_resume(st["id"], a)
    assert [s["status"] for s in new["plan"]] == ["done", "done", "pending"]


def test_resume_mid_write_restores_checkpoint_and_reruns(workspace):
    st = S.new_task(workspace, "fix add", plan=_plan())
    original = Path(workspace, "calc.py").read_text(encoding="utf-8")
    r = C.edit_file(workspace, "calc.py", "return a - b", "return a + b", task_id=st["id"])
    # Interrupted after the checkpoint but the recorded "after" state never matched (partial write).
    Path(workspace, "calc.py").write_text(original.replace("return a - b", "return a +"), encoding="utf-8")
    from backend.agent import checkpoints as CP

    CP.record_after(r["checkpoint_id"], "calc.py", S.fingerprint(workspace, ["calc.py"])["calc.py"])
    st["plan"][0]["status"] = "done"
    st["plan"][1].update(status="running", checkpoint_id=r["checkpoint_id"], after_hashes={"calc.py": "not-this"})
    st["last_checkpoint"] = r["checkpoint_id"]
    S.save(st, status=S.INTERRUPTED)
    a = S.analyze_resume(st["id"])
    assert a["rollback_checkpoint"] == r["checkpoint_id"] and a["resume_step"] == "apply_fix"
    S.apply_resume(st["id"], a)
    assert Path(workspace, "calc.py").read_text(encoding="utf-8") == original


def test_terminal_tasks_are_not_resumable(workspace):
    st = S.new_task(workspace, "x", plan=_plan())
    S.save(st, status=S.COMPLETED)
    assert S.analyze_resume(st["id"])["resumable"] is False
