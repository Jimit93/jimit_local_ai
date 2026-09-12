# Evaluation report — LinkedIn writer, end-to-end through the frontend

**Date** 2026-09-12 · **Checkpoint before changes** `7af8800` · **Model**
qwen2.5-7b-instruct-q4_0 via llama.cpp, CPU only, seed 42

Every number in this report comes from a run. Generation was driven through the
real UI at `http://127.0.0.1:5173` (LinkedIn Writer → Generate); results were
captured from the browser's own network log, not by calling the service directly.

---

## BASELINE (iteration 0)

| | |
|---|---|
| Overall score | **0.6381** |
| Content | 0.500 |
| Style | 0.849 |
| Structure | 0.833 |
| Factual | 0.500 |
| Technical | 0.800 |
| Audience | 0.133 |
| Efficiency | 1.000 |

Validator reported `style_match` **1.00 / SUCCESS** with zero issues.

## FINAL (iteration 3)

| | |
|---|---|
| Overall score | **0.7914** |
| Content | 0.500 |
| Style | 0.849 |
| Structure | 0.833 |
| Factual | **1.000** |
| Technical | 0.733 |
| Audience | **0.733** |
| Efficiency | 1.000 |

Validator reports `style_match` **1.00 / SUCCESS** — and this time it is earned:
`facts_used` passes.

## EFFICIENCY

| | Before | After |
|---|---|---|
| LLM calls | 1 | **1** |
| Tokens | 956 | 993 |
| Latency | 45.3 s | 47.2 s |
| Retrieval count | 2 | 2 |

Cost is essentially unchanged: +37 tokens and +1.9 s for the longer facts framing.
The design target of one model call per generation is preserved. Iteration 2
temporarily broke this (2 calls, 1230 tokens, 57.6 s) and iteration 3 restored it.

## FAILURE FOUND

**Two failures, both below the model.**

**F1 — VALIDATION (with PROMPT contributing).** 18 measured numbers were supplied
and the post cited none. The validator could not see this: `numbers_grounded` only
checks that numbers *used* are permitted, so a post containing no numbers satisfies
it vacuously and scores 100%. The prompt compounded it — three restrictive
instructions about numbers ("do not add others", "if a number is not provided, do
not state one", "never invent metrics") and not one obligating instruction.

**F2 — HARNESS `build_plan` (with PROMPT rendering).** The post ended "Hiring for
more engineers with a practical approach to AI" while the audience was hiring
managers — the author's intent inverted. Two causes: `build_plan` forced a CTA even
though `cta_expected` is False (measured 20.8% usage, threshold 40%), and the
internal detector label `hiring` was passed to the model verbatim. That label is a
misnomer — it matches "open to", "opportunities" and "collaboration", and of the 9
corpus posts it fires on, **0 announce a vacancy**.

**Neither was the transformer's fault.** The prompt was reconstructed offline and
matched the API-reported length byte for byte (3,311 chars), proving the model
received both the facts and the misleading CTA instruction.

## FIX

| # | Layer | Change |
|---|---|---|
| 1 | HARNESS + PROMPT rendering | `build_plan` falls back to the author's usual CTA only when `cta_expected`; new `style_engine.CTA_GUIDANCE` translates the label (mirrors the existing `HOOK_GUIDANCE`); `writer_prompt` renders guidance text |
| 2 | VALIDATION | `validate_post` gained a deterministic `facts_used` check, added to `CRITICAL_CHECKS` |
| 3 | PROMPT + HARNESS | One obligating constraint, emitted only when facts exist; the repair prompt now receives the facts as `context=` |

No new dependency. No architecture change. No LLM added anywhere.

### What iteration 2 cost, and why it was kept

Iteration 2 scored **worse** overall (0.6791 → 0.6599). The gate fired the existing
repair pass, which cost a second model call (+289 tokens, +13.4 s) and returned
text identical to the input. That failure was the most valuable result of the whole
exercise: it exposed that `linkedin_writer` built the repair prompt **with no
`context=`**, so the facts were never shown to the repair call. It was being told to
cite numbers it could not see. Iteration 3 fixed both that and the root cause.

## REGRESSION

| | |
|---|---|
| Pre-existing tests before | 123 passed, 2 skipped |
| Pre-existing tests after | 123 passed, 2 skipped |
| New regression tests | **43 passed** (`tests/regression/`) |
| Total | **166 passed, 2 skipped** |
| Frontend typecheck | `tsc -b` clean |
| Failures | none |

New suites: `test_linkedin_audience_001` (CTA not forced, label translated, no
inversion), `test_linkedin_facts_001` (evidence gate catches and passes correctly,
repair carries facts), `test_feedback_routing_001` (feedback reaches the right
layer, never calls the model).

No increase in LLM calls, no change to routing, retrieval, the database, the
corpus, or unrelated UI.

## DETERMINISM

Generation is seeded (`SETTINGS.llm.seed = 42`, passed to llama.cpp). A repeat run
of iteration 3 through the frontend returned **byte-identical** output. These
scores therefore carry no sampling noise — but they describe **one brief**. Nothing
here establishes that the fixes generalise to other topics.

## CONCLUSION

**The improvement is real but narrower than the headline number suggests.**

What genuinely improved, and is protected by deterministic tests:

- The **factual inversion is gone**. A post aimed at hiring managers no longer
  announces that the author is recruiting. This was a correctness bug that would
  have actively damaged the user's stated goal.
- **Supplied evidence is now used** — 3 of 18 numbers cited on the first call —
  and, more importantly, its absence is now *detectable*. Before, an evidence-free
  post scored a perfect 100%; the metric was lying.
- Both fixes cost **zero extra model calls**.

What did not improve, stated plainly:

- **Content score never moved** (0.500 at every iteration).
- **Technical score fell** 0.800 → 0.733, and audience reached only 0.733, not 1.0.
- The post still opens with the wrong hook type (`what_if`, planned `claim`), still
  runs to a single paragraph against a corpus median of 2, and still appends
  generic hashtags that are not the author's own.
- Only 3 of 18 numbers were used, and not the most persuasive ones — the −82.6%
  reduction in LLM calls is arguably the headline figure and was not chosen.

The overall +0.1533 (+24% relative) is carried almost entirely by two components.
That is an honest result, not a broad quality improvement, and the remaining
defects are recorded in `docs/feature_backlog.md` rather than papered over.

**Fine-tuning is not warranted.** Both failures were fixed below the model, and
iteration 3 shows Qwen complies once the instruction balance is corrected. There is
no evidence the weights are the limiting factor.
