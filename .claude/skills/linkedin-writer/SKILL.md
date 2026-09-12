---
name: linkedin-writer
description: Generate or revise a LinkedIn post in the user's measured voice using the local style engine. Use when asked to write, draft, improve, or critique LinkedIn content, or to explain what makes the user's writing distinctive.
---

# LinkedIn writer

The style profile is **measured**, not guessed. Never invent style rules; read them
from `config/style_profile.json` (or `style_engine.style_instructions()`).

## Pipeline (do not reorder)

1. `style_engine.load_profile()` → measured distributions.
2. `style_engine.style_instructions(profile, target_words)` → imperative rules.
   **Never pass raw percentages to the model** — it writes them into the post.
3. `linkedin_writer.select_examples(topic, k=2)` → 1–3 capped examples, deduped.
   Examples are *style* evidence. Their factual claims are history, not today.
4. `prompt_builder.writer_prompt(...)` → one compact prompt.
5. **One** `LLM.generate` call.
6. `_strip_wrapper` → `_enforce_length` → `validator.validate_post`.
7. At most **one** repair call, listing only the checks that actually failed.

## Facts rule

Facts come only from the `facts` argument supplied for this request. If a number
was not given, do not state one. An old post claiming "14 tok/s" is evidence of how
the author writes about metrics, not evidence of current performance.

## What the profile contains

Length percentiles, paragraph and sentence rhythm, hook archetypes (claim, metric,
label, greeting, gratitude, story, announcement, contrarian, question, listicle,
what_if, transformation), CTA types (comment, dm, link, follow, hiring), hashtag and
emoji rates, technical density, recurring phrases, vocabulary and verb preferences,
banned generic phrases.

## Quality bar

`validate_post` must reach SUCCESS (style_match ≥ 0.85). Common failures and their
deterministic fixes:

| Failure | Fix |
|---|---|
| Title line / `Claim:` / `Greeting:` labels | `_strip_wrapper` |
| 2× over length | `_enforce_length` (drops trailing paragraphs, keeps hashtag tail) |
| `[Author's Name]` placeholders | validator flags → repair call |
| Generic filler ("game-changer", "delve into") | banned-phrase check → repair call |
| Copied sentence from an example | overlap check ≥ 0.6 → regenerate |

## Model note

Measured on this machine: the 1.8B needs all of the above scaffolding and still
runs long. Set `JLA_MODEL_PREFER=large` for the 7B when quality matters more than
the ~3× latency.
