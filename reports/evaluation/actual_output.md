# Actual output — iteration 3 (final)

Same brief, same frontend form, same two retrieved style examples (doc 5809, 5807)
as the baseline. Generated through the UI, captured from `POST /api/writer/generate`
(network request `10312.385`).

## Generated post

> What if your AI agent could route tasks to the cheapest processor that can handle
> them, making the language model the last resort? On a benchmark run of 23 tasks,
> the latency dropped by 83.0% and CPU time by 81.5%. This makes it perfect for
> running entirely locally and offline, even on low-end CPUs.
>
> #AI #Engineering #TechInnovation

## Actual profile vs predicted

| Dimension | Predicted (from corpus) | Actual | Verdict |
|---|---|---|---|
| Length | 38–137 words (median 64) | 60 words | in band |
| Paragraphs | 1–3 | 1 | in band, but no closing line |
| Avg sentence length | ~19 words | 15 (max 23) | in band |
| Hook | claim (58% of posts) | **what_if** | mismatch |
| Technical density | floor 0.02 | 0.1833 | above floor |
| Hashtags | 0–5 | 3 | in band |
| Emoji | sparing, allowed | 0 | fine |
| CTA | not expected (21% of posts) | none | correct |
| Evidence cited | ≥2 of 18 numbers | **3** (23, 83.0, 81.5) | pass |

## Execution metadata

| Metric | Value | Baseline |
|---|---|---|
| Style match | 100% (honest) | 100% (misleading) |
| Status | SUCCESS | SUCCESS |
| `facts_used` | **true** | check did not exist |
| Validator issues | none | none |
| LLM calls | **1** | 1 |
| Tokens | 993 | 956 |
| Prompt tokens | 913 | 884 |
| Latency | 47.2 s | 45.3 s |
| Repair pass | not needed | not applicable |
| Sources retrieved | 2 | 2 |

## Scored result

| Component | Baseline | Final | Δ |
|---|---|---|---|
| Content | 0.500 | 0.500 | 0.000 |
| Style | 0.849 | 0.849 | 0.000 |
| Structure | 0.833 | 0.833 | 0.000 |
| Factual | 0.500 | **1.000** | **+0.500** |
| Technical | 0.800 | 0.733 | −0.067 |
| Audience | 0.133 | **0.733** | **+0.600** |
| Efficiency | 1.000 | 1.000 | 0.000 |
| **Overall** | **0.6381** | **0.7914** | **+0.1533** |

## What is still not right

Honest list, none of it fixed:

- **Hook mismatch.** The plan asked for `claim`; the model opened with `what_if`.
  Nothing in the system verifies the planned hook against the produced hook, so
  this is invisible to the validator. Queued in the backlog.
- **Single paragraph.** The corpus median is 2. The post reads as one block plus a
  hashtag line.
- **Generic hashtags.** `#AI #Engineering #TechInnovation` are not the author's own
  tags, and the plan asks for up to 5 although the corpus median is 0.
- **Content score flat at 0.500** across every iteration — it measures coverage of
  12 expected topic keywords and no fix targeted it.
- **Only 3 of 18 numbers used.** The gate requires 2; it does not push toward the
  most persuasive subset (the −82.6% LLM-call reduction is arguably the headline
  number and was not chosen).
