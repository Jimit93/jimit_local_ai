# Refine-loop backtest

Run `a3edbfd371` · model `qwen1_5-1_8b-chat-q4_k_m.gguf` · seed 42 · 5 briefs · 294.3 s

**Read the columns correctly.** *Refine draft → Refine loop* is paired: the same draft before and after refinement. *One-shot* is a different sample - the seed is applied once at model load, so it compares the two paths end to end, not as a controlled test.

| Metric | One-shot | Refine draft (before fixes) | Refine loop |
|---|---|---|---|
| Pass rate (SUCCESS) | 0.2 | 0.0 | 1.0 |
| Complete ending | 1.0 | — | 1.0 |
| Has hashtags | 1.0 | — | 1.0 |
| Mean overall score | 0.7316 | 0.7148 | 0.8515 |
| Mean critical failures | 1 | — | 0 |
| Posts with banned phrases | 3 | — | 0 |
| Mean LLM calls | 1.8 | — | 1 |
| Mean tokens | 1529.8 | — | 982.4 |
| Mean latency (ms) | 37064.1 | — | 21755.9 |

## Score by micro-iteration (mean across briefs)

0.7148 → 0.7779 → 0.8035 → 0.8515

Last iteration that still improved the mean: **3** of up to 20 allowed (deepest loop used 3).

## What the checks could not see (read from the generated text)

- Every post passed, and several would still be rejected by a human reader. The loop optimises against the validator, so the validator's blind spots now decide quality.
- Topic drift passes: harness_proof asked for a routing-harness post for hiring managers and got 'Are you a CEO dealing with manual operations...', lifted from retrieved example doc 5809. No check compares the post with the topic.
- Retrieved-example content reused as fact passes: 'The AI Business Accountant, designed by our team' (business_automation_no_facts) is the author's own product from doc 5807.
- Biography pulled from old posts passes: 'an AI engineer and quant developer at IIT Madras' (cpu_inference one-shot) and 'my recently completed Advanced Generative AI Master's program' (cpu_inference refine). Rule 8 says facts come only from the brief.
- Generic register passes: 'In the realm of', 'has never been more critical', 'ever-evolving digital landscape'.
- The loop can lower the score while clearing a defect: business_automation_no_facts went PARTIAL_SUCCESS 0.8502 -> SUCCESS 0.7642 because the drift guard ranks critical defects above score, and dropping two hype sentences also removed content.
- '20x' did not hold here: the mean score stopped improving at iteration 3 of 20, and 0 of 15 budgeted model fixes were needed. Iteration ran out of defects it could see, not out of room to improve.

## Per brief

- **harness_proof** — one-shot PARTIAL_SUCCESS 0.6057 (2 calls) → refine SUCCESS 0.8193 (1 calls, 3 iterations, stopped: passed)
- **business_automation_no_facts** — one-shot PARTIAL_SUCCESS 0.7693 (2 calls) → refine SUCCESS 0.7642 (1 calls, 2 iterations, stopped: passed)
- **cpu_inference** — one-shot SUCCESS 0.8106 (1 calls) → refine SUCCESS 0.8001 (1 calls, 2 iterations, stopped: passed)
- **deterministic_validator** — one-shot FAILURE 0.6112 (2 calls) → refine SUCCESS 0.9368 (1 calls, 2 iterations, stopped: no_fixer_could_improve)
- **rag_small_corpus** — one-shot PARTIAL_SUCCESS 0.861 (2 calls) → refine SUCCESS 0.9369 (1 calls, 2 iterations, stopped: passed)
