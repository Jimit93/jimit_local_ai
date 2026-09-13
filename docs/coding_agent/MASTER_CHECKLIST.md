# Coding agent — master checklist

Persistent. Updated after every phase. `[x]` = done **and verified**; `[~]` = partial,
with the reason; `[ ]` = not started.

## Phase 0 — understanding and inspection
- [x] Prompt understood (requirements extracted into PLAN.md §3–5)
- [x] Existing environment inspected (PLAN.md §2)
- [x] Existing project inspected
- [x] Existing DeepSeek Harness located — **does not exist**; existing routing harness is the orchestrator
- [x] Existing architecture understood
- [x] Existing tools identified
- [x] Existing memory identified (`agent_memory`)
- [x] Existing tests identified (166 pass, 2 skip)
- [x] Existing frontend identified (8 pages)
- [x] Existing backend identified
- [x] Existing models identified
- [x] Missing functionality identified (capability matrix)
- [x] Architecture designed
- [x] Resource governor designed
- [x] Routing designed

## Build
- [x] Tool system validated (Phase 1, `19a87f1`) — 23 workspace tools in the registry
- [x] Safety policy + audit (Phase 1) — sandbox, allowlist, redaction, approvals
- [x] Resource governor implemented + measured (Phase 2, `670b4a6`) — see reports/agent/cpu_*.json
- [x] RAM reclaim (user request 2026-09-13) — tiered, protected processes never touched
- [x] NLP layer for code (Phase 3, `89c4e4f`) — fused P@1 0.912 / MRR 0.949 (reports/agent/ranking_eval.json)
- [~] ML layer — intent model pending (Phase 5); learned ranker not adopted: fused BM25 already MRR 0.949
- [x] Context system (Phase 3) — symbol excerpts, budget, redaction
- [x] Memory (Phase 4) — session/project/task/persistent
- [x] Persistent state (Phase 4)
- [x] Checkpoint system (Phases 1+4) — conflict-aware rollback
- [x] Resume system (Phase 4) — unit-tested; real restart backtest pending (Phase 9)
- [ ] LLM layer: code tiers (Phase 5)
- [ ] Harness integration (Phase 5)
- [ ] Coding agent loop (Phase 6)
- [ ] Validation loops (Phase 6)
- [ ] Recovery system (Phase 6)
- [ ] API + worker (Phase 7)
- [ ] Frontend (Phase 8)

## Test and evaluate
- [ ] Backend tested
- [ ] Frontend tested
- [ ] Integration tested
- [ ] E2E tested
- [ ] CPU tested (idle / normal / heavy / peak / protection / recovery)
- [ ] Memory (RAM) tested
- [ ] Restart tested
- [ ] Stress tested
- [ ] Backtest completed
- [ ] Predicted vs actual evaluated
- [ ] Graphs generated

## Improve
- [ ] Missing features audited
- [ ] Before baseline recorded
- [ ] Improvements implemented
- [ ] Before/after comparison
- [ ] Regression testing completed
- [ ] Final frontend/backend/E2E backtest completed

## Close
- [ ] Documentation (architecture + user guide)
- [ ] Security audit completed
- [ ] Final system audit completed

## Measured facts that changed decisions
- Machine on battery: max achievable system CPU ≈ 68%; protection validated at a lowered
  cap (activation 0.0 s after first sample above cap, recovery 0.06 s). Re-run plugged in.
- llama.cpp 10 threads = 22.5 tok/s @ 58.5% vs 16 threads = 22.1 tok/s @ 67.5% → default threads from cap.
- Embeddings for code ranking not adopted (MiniLM load ≈ 20 s; BM25 fused MRR 0.949).

## Improvement candidates (designed, NOT implemented — need a baseline first)

Source: user-supplied "AI App Development Lifecycle & Rapid Iteration" notes (Describe →
Generate → Test → Refine; many small iterations beat one big prompt).

- **I1 Micro-iterations for medium/high-complexity edits.** Today one task = one edit proposal
  for the whole request (the "one big prompt" failure mode for a 3B model). Plan: split into
  <= 5 increments (deterministic first: explicit list items, "and also", one file per increment;
  small-model plan only if that fails), each with its own checkpoint + propose/apply/verify/repair;
  a failure rolls back and repairs only that increment. Not 20×: each model iteration costs
  ~30–60 s CPU here. Low-complexity fixes stay single-cycle.
  Measure: multi-part feature tasks, single-cycle (baseline) vs incremental — success rate,
  LLM calls, tokens, wall time.
- **I2 Human Refine step.** A follow-up in the same session ("now also…", "make it…") continues the
  previous task's files, diff and checkpoint chain instead of starting cold.
  Measure: follow-up success and context tokens vs a fresh task.

## Open user requests
- Latest coder models: qwen3-coder 30b (19 GB) does not fit 16.7 GB RAM; deepseek-coder-v2 16b
  lite (8.9 GB) fits only after RAM reclaim; qwen2.5-coder 1.5b/3b/7b fit. **Download needs user OK.**

## State
- Current phase: 5 (intents + LLM code tiers + harness integration)
- Last checkpoint: phase-4 commit on `feature/coding-agent`
- Tests: 83 agent tests pass; full suite last run 209 passed / 2 skipped (after phase 1)
- Next action: `backend/agent/intents.py`, code tiers in `llm/manager.py`, router `workspace`/`coding` chains
