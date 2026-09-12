# LinkedIn style back-test

6 held-out posts, 57 posts used for the style profile (seed 7). Evaluation posts were excluded from both the profile and example retrieval; leakage checks found **0** violations.

## Aggregate

| Measure | Value |
|---|---|
| length similarity | 0.7999 |
| sentence len similarity | 0.4569 |
| paragraph similarity | 0.5889 |
| technical density similarity | 0.5783 |
| readability similarity | 0.4441 |
| vocabulary overlap | 0.121 |
| topic term recall | 0.2267 |
| hook match rate | 0.5 |
| cta match rate | 0.8333 |
| hashtag behaviour match rate | 0.5 |
| verbatim copy rate | 0.0 |
| mean style score | 0.578 |
| mean llm calls | 1.0 |
| mean tokens | 807.7 |
| mean latency ms | 19618.1 |

## Per post

| Post date | Style score | Words orig→gen | Hook orig→gen | CTA match | Vocab overlap | Verbatim copy |
|---|---|---|---|---|---|---|
| 2025-08-05 | 0.3673 | 69→104 | gratitude→label | yes | 0.1463 | no |
| 2023-11-24 | 0.6949 | 45→51 | claim→claim | yes | 0.1163 | no |
| 2025-10-18 | 0.657 | 212→204 | claim→claim | yes | 0.141 | no |
| 2025-07-19 | 0.5787 | 52→64 | metric→label | yes | 0.0714 | no |
| 2023-12-31 | 0.7766 | 48→59 | claim→claim | yes | 0.1628 | no |
| 2026-08-05 | 0.3932 | 145→218 | label→milestone | no | 0.0882 | no |

**Reading this table:** vocabulary overlap is expected to be low - the aim is reproducing structure, rhythm and habits, not wording. A `YES` in the verbatim column is a failure and halves that post's score.
