---
name: coding-agent
description: Extend or debug the local coding agent - add an intent or tool, change a plan or the loop, investigate a failed/blocked task, run the agent backtests. Use when working in backend/agent/, the /api/agent routes or the Coding Agent UI pages.
---

# Coding agent

Architecture and every contract: `docs/coding_agent/ARCHITECTURE.md`. Read the section for the
part you are changing first.

## Debug a failed or blocked task

1. Read its record, not the UI: `python <scratch>/diag` style query on `agent_tasks.state_json`
   (plan statuses, `last_output`) and `agent_events` (kinds `retry`, `repair`, `governor`, `ram`,
   `rollback`, `stop`). Backtests keep their own DB under `data/backtest/`.
2. Classify the cause before changing code:
   * `InsufficientRAM` → environment (another process holds RAM), not a code bug.
   * `proposed edit is not applicable` twice → the model output; check `data.proposal.raw`,
     the allowed files and the context the step received.
   * a read-only plan for an edit request → intent rules (`intents.py`), add the phrasing to
     `tests/agent/test_agent_intents.py`.
   * wrong files edited or offered → `h_locate` / `functions_under_test` / ranking signals.
3. Fix the root cause, add a test that reproduces it, re-run the one backtest case:
   `python -m scripts.agent_backtest --cases <id> --label debug_<id>`.

## Add a deterministic intent

1. Rule in `intents.RULES` (specific before general; build patterns by string concatenation,
   never rf-strings - `{0,30}` becomes "(0, 30)"), plus seed phrasings in `SEED`.
2. Add it to `DETERMINISTIC` and implement it in `executor._dispatch` returning `_res`/`_fail`.
3. Tests: phrasing in `test_rules`, execution through `run_intent_tool`, and the classifier
   snapshot test must stay green (`test_existing_classifications_are_unchanged`).

## Add a tool

Function in `code_tools.py` (workspace first, `policy.resolve_path`, checkpoint before any write),
register it in `tools/registry._bootstrap_workspace`, test it in `tests/agent/test_agent_code_tools.py`.
New commands need an allowlist entry in `policy.check_command` with a risk level.

## Change the loop or a plan

* Plans are data (`planner.py`); handlers are `h_<kind>` in `loop.py` returning `Outcome`.
* Repairs return to `REPAIR_TARGET[kind]`; keep rollback-before-repropose.
* Any new model call goes through `_llm_call` (cache, governor admission, tier fallback,
  completion `finish` mode) and its prompt goes in `harness/prompt_builder.py`.
* Run `tests/agent/test_agent_loop.py` (scripted model, real tools) before a backtest.

## Measure

Never quote a number that did not come from a run. The dashboard reads `reports/agent/*.json`.
Compare before/after on identical case sets; record free RAM and power state with the run
(the laptop on battery tops out near 68 % system CPU).
