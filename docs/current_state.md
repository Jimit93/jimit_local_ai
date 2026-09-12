# Compact state

Read this first. Inspect only the files the current task names.

## Current task

QA evaluation of the LinkedIn writer through the real frontend. See
[active_task.md](active_task.md).

## Completed milestones

- Model runtime works — qwen2.5-7b-instruct-q4_0 via llama.cpp, warm in ~37 s.
- Ingestion works — 6,143 docs; 136 posts; 77 usable for style.
- Retrieval works — 127 chunks / 127 embeddings, chunk-level BM25 + MiniLM fusion.
- Harness works — 125 tests pass, 2 skipped (opt-in inference).
- Frontend works — Vite on :5173, proxies `/api` to :8077.
- Baseline benchmark works — run `b79f795399`: LLM calls −82.6%, tokens −73.5%,
  latency −83.0%, accuracy 39.1% → 100%.
- Evaluation layer works — `backend/benchmarks/evaluation.py`, baseline scored.

## Failures found and FIXED this task

| # | Layer | Problem | Fix |
|---|---|---|---|
| F1 | VALIDATION (+ prompt) | 18 numbers supplied, 0 cited; `numbers_grounded` passed vacuously at style_match 100% | `facts_used` check in `CRITICAL_CHECKS`; one obligating prompt constraint |
| F2 | HARNESS `build_plan` (+ prompt rendering) | CTA forced despite `cta_expected` False; label `hiring` inverted intent (0 of 9 such posts recruit) | Respect `cta_expected`; `CTA_GUIDANCE` translates the label |
| F3 | HARNESS `linkedin_writer` | Repair prompt had no `context=`, so fact issues were unfixable | Repair prompt now carries the facts |

Result: overall 0.6381 -> 0.7914, still one LLM call. Generation is seeded
(seed 42) so runs are byte-reproducible; the caveat is one brief, not noise.

## Open weaknesses (not fixed, queued)

Hook type never verified against the plan; hashtag target uses p75 against a
corpus median of 0; content score flat at 0.500; single-paragraph output.

## Files changed this task

See the table in [active_task.md](active_task.md).

## Tests

166 passed, 2 skipped (123 pre-existing unchanged + 43 new in `tests/regression/`).
Frontend `tsc -b` clean.

## Next action

Task complete. Highest-value backlog item: verify the planned hook against the
generated hook.

## Backlog summary

14 items queued in [feature_backlog.md](feature_backlog.md). Nothing outside the
current task has been implemented.
