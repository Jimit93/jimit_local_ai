# System audit

Produced during the QA evaluation task (2026-09-12). Scope note up front: this
audit is based on the components the task actually touched — the harness, the
validator, the style engine, the writer, the prompt builder, the retrieval
interface, the DB schema, the API surface, and the Writer / Benchmarks / App
frontend. **Not audited in depth**: the ingestion internals, chunking, the LLM
adapters, and the Chat / Knowledge / Tasks / System / Settings pages. Items
resting on those are marked *(unverified)*.

---

## Already working

| Capability | Evidence |
|---|---|
| Cheapest-processor routing | Benchmark `b79f795399`: 19 of 23 tasks answered with no LLM call; calls −82.6%, tokens −73.5%, latency −83.0%, accuracy 39.1% → 100% |
| `ProcessorResult` envelope everywhere | Enforced by `tests/test_router.py`; every processor returns it |
| LLM is last in every chain | `CHAINS` in `router.py`; `p_llm` terminal or absent |
| Single model instance | `llm/manager.py` singleton; `warmup()` in the lifespan hook |
| Style profile is measured, not asserted | `config/style_profile.json` built from 77 posts; percentages traced to the corpus |
| Statistics kept out of prompts | `style_instructions()` emits imperative rules; verified in the reconstructed prompt |
| Deterministic output cleanup | `_strip_wrapper` / `_enforce_length` remove model scaffolding with no second LLM call |
| Deterministic QC | `validate_post` — 11 checks, no model call |
| Chunk retrieval, document-level style | `select_examples` uses `Hit.doc_text`; both runs retrieved doc 5809/5807 consistently |
| Facts vs style separation | Historical posts are labelled as history in the prompt; `numbers_grounded` blocks invented figures |
| Local-only operation | Bind is 127.0.0.1; HF offline forced; no outbound calls observed in any run |
| Evaluation layer (new) | `backend/benchmarks/evaluation.py` + `/api/evaluation/report`, deterministic |
| Feedback routing (new) | `backend/services/feedback.py`, 11 phrase classes, no model call |

## Fixed during this task

| Defect | Layer | Fix |
|---|---|---|
| CTA forced on despite `cta_expected` False (20.8%) | HARNESS `build_plan` | Fall back to `preferred_ctas[0]` only when the author actually uses a CTA |
| Internal label `hiring` reached the model as instruction text, inverting intent (0 of 9 such posts recruit) | PROMPT rendering | `style_engine.CTA_GUIDANCE` + `cta_guidance()`, mirroring the existing `HOOK_GUIDANCE` |
| Evidence-free post scored 100% / SUCCESS | VALIDATION | `facts_used` check, added to `CRITICAL_CHECKS` |
| Repair pass could never fix a fact issue | HARNESS `linkedin_writer` | Repair prompt now receives the facts as `context=` |
| Facts supplied but never cited | PROMPT | One obligating constraint, emitted only when facts exist |

## Missing

| Gap | Value | Risk | Effort | Dependencies |
|---|---|---|---|---|
| Planned hook is never verified against the produced hook (planned `claim`, got `announcement` then `what_if`, silently) | HIGH | low | small | pass `plan` into `validate_post` |
| No retrieval-quality measurement (precision@k); "was retrieval at fault?" cannot be answered numerically | HIGH | low (read-only) | medium | a small labelled query set |
| Feedback is stored but nothing consumes it — no loop from `agent_memory` back into constraints | MEDIUM | medium | medium | the feedback store (now exists) |
| No per-request override of retrieval `top_k` or layer thresholds in the UI | LOW | low | small | Settings page |
| Ingestion duplicate-detection behaviour *(unverified)* | UNKNOWN | — | — | needs an ingestion audit |

## Weak / needs improvement

| Weakness | Detail |
|---|---|
| `build_plan` uses p75 for hashtags | Asks for up to 5 while the corpus median is 0 and only 36% of posts use any. Same class of bug as the CTA one, deliberately left out of this task so the before/after stayed attributable. |
| `content_score` flat at 0.500 | Keyword-coverage proxy; never moved across four runs. Either the metric or the generation needs work — currently it measures little. |
| Single-paragraph output | Corpus median is 2 paragraphs; three of four runs produced 1. |
| Generic hashtags | `#AI #Engineering #TechInnovation` are not the author's own tags. |
| RAM headroom | Auto-selection loaded the 7B model at 7.9 GB RSS, leaving 0.38–1.13 GB free. A second process can thrash. |
| Frontend health polling | `App.tsx` polls `/api/health` every 6 s and each poll runs `psutil`; cheap individually, wasteful on a CPU-bound box. |
| Evaluation is n=1 | Every score comes from a single sampled generation at temperature 0.75. |

## Unnecessary / redundant

- Nothing found that should be deleted. The one thing that *looked* redundant — the
  repair pass — turned out to be broken rather than superfluous, and is now
  functional but rarely needed (iteration 3 avoids triggering it).

## Recommended next features

Ordered by value ÷ (complexity × regression risk).

1. **Hook verification** — VALUE high, RISK low, EFFORT small, DEPS none.
   The plan already picks a hook; add the check. Purely deterministic.
2. **Hashtag target from usage rate, not p75** — VALUE medium, RISK low, EFFORT
   small, DEPS none. Identical shape to the CTA fix already shipped.
3. **Retrieval precision@k harness** — VALUE high, RISK low (measurement only),
   EFFORT medium, DEPS a labelled query set. Without it, retrieval remains the one
   upstream layer that cannot be blamed or exonerated with a number.
4. **Repeat each evaluation n=3 and report the spread** — VALUE high (it is the
   main threat to every conclusion in this report), RISK none, EFFORT small but
   slow to run (~45 s per generation), DEPS none.
5. **Feedback → constraint loop** — VALUE medium, RISK medium, EFFORT medium,
   DEPS the feedback store. Only worth building once several real feedback items
   exist; building it now would be speculative.

## Features intentionally not implemented

- **An LLM judge for post quality.** It would cost a model call on a CPU-only box
  to measure what the deterministic validator measures for free, and it is exactly
  the escalation the harness exists to prevent.
- **Fine-tuning Qwen.** Both measured failures were below the model and were fixed
  deterministically. Iteration 3 shows the model complies when the instruction
  balance is right. There is no evidence that weights are the limiting factor.
- **A second evaluation store.** `agent_memory`, `task_runs` and `benchmark_runs`
  already exist; adding a store would violate the no-duplicate-systems rule.
- **Dark mode / cosmetic UI work.** Out of scope, no user request.
