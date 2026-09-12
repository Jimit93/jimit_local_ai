---
name: benchmarking
description: Run or extend the baseline-vs-harness benchmark and the LinkedIn style back-test. Use when measuring LLM-call, token, latency or CPU reduction, or when adding benchmark tasks.
---

# Benchmarking

## Rules

1. **Never write a number you did not measure.** Not in a report, not in a README,
   not in a commit message.
2. Ground truth is computed at run time (`tasks.py`: `_count()`, `_keyword_hits()`,
   `_word_count()`), never typed by hand. A hand-written expected value was wrong
   once already — it cost a debugging cycle.
3. Baseline and harness must use the **same model in the same process**. Pin it with
   `--model small|large` so two runs are comparable.
4. Cold start (model load, MiniLM load, index build) is paid in `warmup()` before
   any task is timed, and reported separately.

## Grading modes

| Mode | Meaning |
|---|---|
| `exact` | answer must contain the independently computed value |
| `set` | every required token must appear |
| `soft` | keyword-coverage heuristic — a score, never called "correctness" |
| `style` | the deterministic style validator, applied to both modes equally |

## Running

```
run_benchmark.bat                      full, with baseline (minutes on CPU)
python scripts/run_benchmark.py --no-baseline --suites deterministic nlp
run_backtest.bat                       held-out LinkedIn style reproduction
```

Outputs: `reports/benchmark_results.json`, `reports/benchmark_report.md`,
`reports/linkedin_backtest.{json,md}`, plus rows in `benchmark_runs`.

## Back-test leakage

The evaluation posts must be excluded from **both** the style profile
(`build_profile(exclude_doc_ids=...)`) **and** example retrieval
(`select_examples(exclude_ids=...)`). The report carries a `leakage_detected`
count — if it is not 0, the result is void.

High verbatim overlap is a **failure**, not a success: the target is style
reproduction, not memorisation. A copied 6-gram halves that post's score.

## Interpreting

`total_cpu_ms` sums all threads, so it exceeds wall time on a 16-thread run; compare
it between modes, not against the clock.
