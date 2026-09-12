# Active task lock

**Objective** — End-to-end QA of the LinkedIn generation path through the real
frontend: baseline → predicted profile → compare → localize failure → minimal fix
→ retest → regression → benchmark → graph.

**Status** — COMPLETE. Three iterations run, both localized failures fixed, no
regression. Report at `reports/evaluation/evaluation_report.md`.

## Result

| | Baseline | Final |
|---|---|---|
| Overall | 0.6381 | **0.7914** |
| Factual | 0.500 | **1.000** |
| Audience | 0.133 | **0.733** |
| LLM calls | 1 | 1 |

## Files modified

| File | Change |
|---|---|
| `backend/services/style_engine.py` | `CTA_GUIDANCE` + `cta_guidance()` |
| `backend/services/linkedin_writer.py` | `build_plan` respects `cta_expected`; repair prompt carries facts |
| `backend/harness/prompt_builder.py` | CTA rendered as guidance; one obligating evidence constraint |
| `backend/harness/validator.py` | `facts_used` check, added to `CRITICAL_CHECKS` |
| `backend/benchmarks/evaluation.py` | NEW — deterministic predicted-vs-actual evaluator |
| `backend/services/feedback.py` | NEW — feedback → layer classifier |
| `backend/api/app.py` | `/api/evaluation/report`, `/api/evaluation/score`, `/api/feedback` |
| `frontend/src/pages/Evaluation.tsx` | NEW — evaluation panel |
| `frontend/src/components/EvalGraphs.tsx` | NEW — dependency-free SVG charts |
| `frontend/src/pages/Writer.tsx` | feedback controls |
| `tests/regression/` | NEW — 43 tests |

Untouched, as planned: ingestion, chunking, retrieval index, router chains, DB
schema, LLM adapters.

## Test status

- 166 passed, 2 skipped (123 pre-existing + 43 new). Frontend `tsc -b` clean.
- Generation verified through the browser at every iteration.
- Feedback loop verified live: "Doesn't sound like me." → RETRIEVAL, 0.95.

## Next task (not started)

Highest-value backlog item is hook verification — the plan picks a hook type and
nothing checks the result. See `docs/feature_backlog.md`.
