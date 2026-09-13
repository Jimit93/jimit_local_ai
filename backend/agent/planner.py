"""Deterministic plan templates per coding intent.

A plan is a list of steps; each step names its kind (which handler runs it), how many
attempts it gets, whether it modifies files, and what outcome is predicted before it
runs. The loop - not a model - owns sequencing, retries and repair back-edges.
"""
from __future__ import annotations

from backend.agent.intents import IntentResult
from backend.agent.state import make_step

# kind -> the step to return to when this verification step fails (the responsible step)
REPAIR_TARGET = {"verify_syntax": "propose_edit", "verify_tests": "propose_edit",
                 "verify_lint": "propose_edit", "apply_edit": "propose_edit",
                 "verify_requirements": "propose_edit"}
# Intents whose success cannot be judged by the existing tests alone: the requested new
# code must exist (found by hidden tests: a "passing" task had edited the wrong function).
REQUIREMENT_INTENTS = {"implement", "create_file", "write_tests", "refactor"}


def plan_for(ir: IntentResult, *, allow_edits: bool, has_tests: bool) -> list[dict]:
    i = ir.intent
    S = make_step
    if ir.deterministic:
        return [S("run_tool", "tool", max_attempts=1, predicted={"llm_calls": 0, "files_changed": []})]

    read_only = [
        S("locate", "locate", predicted={"llm_calls": 0}),
        S("gather_context", "context", predicted={"llm_calls": 0}),
    ]
    if i == "review":
        return [S("collect_diff", "diff", predicted={"llm_calls": 0}),
                S("static_checks", "static_review", predicted={"llm_calls": 0}),
                S("model_review", "llm_review", critical=False),
                S("report", "report", predicted={"files_changed": []})]
    if i in ("explain_code", "general_coding", "analyze_error") or not allow_edits:
        return read_only + [S("answer", "answer", max_attempts=2), S("check_answer", "validate_answer"),
                            S("report", "report", predicted={"files_changed": []})]
    if i == "rename_symbol":
        steps = [S("locate", "locate"), S("apply_rename", "deterministic_fix"), S("verify_syntax", "verify_syntax")]
        if has_tests:
            steps.append(S("verify_tests", "verify_tests", critical=True))
        return steps + [S("report", "report")]

    steps = []
    if i == "fix_bug" and has_tests:
        steps.append(S("reproduce", "run_tests", predicted={"tests_pass": False}))
    steps += read_only + [
        S("propose_edit", "propose_edit", max_attempts=2),
        S("apply_edit", "apply_edit"),
        S("verify_syntax", "verify_syntax"),
    ]
    if i in REQUIREMENT_INTENTS:
        steps.append(S("verify_requirements", "verify_requirements"))
    steps.append(S("verify_lint", "verify_lint"))
    if has_tests:
        steps.append(S("verify_tests", "verify_tests", critical=True, predicted={"tests_pass": True}))
    steps.append(S("report", "report"))
    return steps
