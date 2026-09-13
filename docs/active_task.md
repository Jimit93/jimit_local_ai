# Active task lock

**Objective** — Implement the Describe → Generate → Test → Refine loop and the
"20x micro-iteration" idea (`Pictures/Structured_PDFs/Category_3_AI_Development_Lifecycle.pdf`)
in the LinkedIn writer, and backtest one-shot vs iterative from the backend and
the frontend.

**Trigger** — A user-pasted draft (buzzwords, invented CEO backstory, cut off at a
colon) was returned as SUCCESS. That is the one-big-prompt failure the PDF
describes, and it shows the loop's Test stage is broken — so Test is fixed first.

**Status** — COMPLETE (2026-09-13). Backtest `a3edbfd371`, report
`reports/evaluation/refine_backtest.md`. 378 passed / 2 skipped (1 deselected: the
coding-agent session's own untracked WIP test). Frontend `tsc -b` clean.

## Result (measured, qwen1.5-1.8B, 5 briefs)

| | One-shot | Refine draft | Refine loop |
|---|---|---|---|
| Pass rate | 20% | 0% | 100% |
| Mean score | 0.732 | 0.715 | 0.852 |
| Model calls | 1.8 | — | 1.0 |
| Tokens | 1530 | — | 982 |
| Latency | 37.1 s | — | 21.8 s |

Draft → loop is paired (same text). One-shot is a different sample (seed applied
once at load). 0 of 15 model fixes used; mean stopped improving at iteration 3/20.
**Validator blind spots remain** (topic drift, example content as fact, biography
from old posts, generic register) — every post passed, not every post is good.

## Milestones

0. **User requirement (added mid-task)**: posts must carry hashtags generated from
   the target-audience parameter. Deterministic NLP, not the model. "Don't change
   or rewrite anything else."
1. TEST stage — validator + deterministic cleanup. *(done, verified on the pasted
   draft: SUCCESS 0.889 → fails on buzzwords + dangling colon; trim restores a
   complete ending; author endings untouched)*
2. REFINE stage — bounded micro-iteration loop, deterministic fixers first,
   budgeted model fixes, reject any pass that lowers the score (drift guard).
3. Backtest — one-shot vs iterative on several briefs, backend runner.
4. Frontend — Writer toggle + iteration trace; Evaluation page backtest chart.
5. Verify through the UI, regression, narrow checkpoint.

## Measured before changing (author corpus, 77 posts)

- Existing banned list hits the author's own posts 1/77 → safe to make critical.
- Candidate phrases with 0/77 author use → safe to ban. Phrases the author does
  use (game-changing, streamline, robust, landscape…) → left alone.
- 56/77 posts end without terminal punctuation → a naive ending check would be
  wrong. Only colon ending in the corpus is the one-word label `Hashtags:`.
- Truncation signal exists: `LLMResult.stop_reason == "length"`.

## Adapting "20x" honestly

20 model calls per post ≈ 5–15 min on this CPU and breaks rule 1. Micro-iterations
are allowed up to 20, but deterministic fixes are free and model fixes are capped.
Whether more iterations help is **measured by the backtest**, not assumed.

## Concurrency — DO NOT TOUCH

Another session is actively building the coding agent on this branch. Its
uncommitted files: `backend/agent/*`, `backend/api/agent_routes.py`,
`backend/benchmarks/agent_eval.py`, `tests/agent/*`, and modifications to
`app.py`, `config.py`, `prompt_builder.py`, `router.py`, `task_classifier.py`,
`llm/manager.py`, `llm/ollama_adapter.py`, `start_backend.bat`.
Since committed as `6a1ec0c`; now also live-editing `frontend/src/App.tsx`,
`frontend/vite.config.ts`, `backend/agent/ram.py`, `backend/api/agent_routes.py`
and new `frontend/src/pages/Agent*.tsx`, `Workspace.tsx`, `agentApi.ts`.
This task's UI therefore goes inside existing pages it owns (Writer, Evaluation)
— no new nav entry, no App.tsx edit.
This task edits none of them except, if unavoidable, a single router include
line in `app.py`. Commits use explicit pathspecs only — never `git add -A`.

## Files this task modifies

`backend/nlp/features.py`, `backend/harness/validator.py`,
`backend/services/linkedin_writer.py`, new `backend/services/refine_loop.py`,
new `backend/benchmarks/refine_backtest.py`, new `backend/api/writer_routes.py`,
`frontend/src/pages/Writer.tsx`, `frontend/src/pages/Evaluation.tsx`,
`frontend/src/services/api.ts`, `tests/regression/`.
