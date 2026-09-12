---
name: harness-routing
description: Add or change a processor in the routing harness - task categories, confidence thresholds, escalation, and the rules that keep work off the LLM. Use when adding a tool, a category, or debugging why a task reached the model.
---

# Harness routing

## Adding a processor

1. Write it as `p_<name>(task: str, ctx: dict) -> ProcessorResult` in
   `backend/harness/router.py`.
2. Return an honest confidence. A processor that cannot answer returns
   `confidence=0.0` **with an `error`** — that is what triggers escalation. Never
   return a plausible-looking guess at high confidence.
3. Put a human-readable answer in `meta["answer"]`; the UI and the benchmark read it.
   An answer that is only a count ("5 results") is not useful — include the content.
4. Add it to the right chain in `CHAINS`, before `p_llm`.
5. Add a test to `tests/test_harness.py` asserting `llm_used is False`.

## Thresholds

`LAYER_THRESHOLD` in `router.py`: deterministic 0.90 > nlp 0.70 > ml 0.60 >
retrieval 0.35 > llm 0.0. Cheaper layers must clear a higher bar, because a wrong
cheap answer is worse than an expensive right one.

## Task categories

`backend/harness/task_classifier.py` — rules first (microseconds, high precision),
then a TF-IDF+LogReg intent model trained on the seed examples in that file.

Rule ordering matters and has bitten before:
- `classification` before `creative_generation` ("classify this LinkedIn post").
- `text_processing` before `structured_data` ("how many words in this text" is not
  a database question) and before `creative_generation` (a URL ending `/post`).

When you add a rule, add its case to `test_intent_rules_do_not_collide`.

## Escalation rules

- Escalate on low confidence or an error, never on "the answer looks short".
- **A correct zero is not a failure.** An exact search finding nothing answers the
  question asked; escalating to semantic search answers a different one. Surface the
  suggestion instead (`suggest_semantic`).
- Escalation count is recorded per task and shown in the UI trace.

## Cost discipline

- `context_manager.build_context` enforces a character budget; blocks are added in
  priority order and dropped when the budget runs out.
- Prefer one good LLM call to several small ones. A second call needs a stated
  reason (the repair call lists the exact failed checks).
- Every LLM call goes through `llm.manager.generate` so `USAGE` stays accurate.
