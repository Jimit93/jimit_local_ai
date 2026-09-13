# refine_on_trial_2

Run at 1789299600 · 67.9 s · power plugged: False · small model qwen2.5-coder:3b

**1/1 cases passed (100%)** · 1 LLM calls · 893 tokens

| Suite | Passed | Pass rate | LLM calls | Tokens | Mean s |
|---|---|---|---|---|---|
| coding | 1/1 | 100% | 1 | 893 | 22.6 |

## Predicted vs actual

- status accuracy 1.0
- LLM budget adherence 1.0
- method accuracy (LLM used iff predicted) 1.0
- files changed F1 (mean) 1.0

## Resources

- CPU avg 43.5% · peak 66.8% · samples above 70%: 0.0
- minimum free RAM 2.24 GB

## Cases

| Case | Suite | Result | s | LLM | Detail |
|---|---|---|---|---|---|
| refine_followup | coding | PASS | 22.64 | 1 | hidden follow-up tests passed=3 failed=0; refine_of=77c4a4b563c5 |
