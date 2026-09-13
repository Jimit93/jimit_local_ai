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
- [ ] Resource governor designed → Phase 2
- [x] Routing designed

## Build
- [ ] Tool system validated (Phase 1)
- [ ] Safety policy + audit (Phase 1)
- [ ] Resource governor implemented + measured (Phase 2)
- [ ] NLP layer for code (BM25 ranking) (Phase 3)
- [ ] ML layer (intent model; ranking only if measured to help) (Phases 3, 5)
- [ ] Context system (Phase 3)
- [ ] Memory (Phase 4)
- [ ] Persistent state (Phase 4)
- [ ] Checkpoint system (Phase 4)
- [ ] Resume system (Phase 4)
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

## State
- Current phase: 1
- Last checkpoint: phase-0 commit on `feature/coding-agent`
- Next action: implement `backend/agent/policy.py` + `code_tools.py`
