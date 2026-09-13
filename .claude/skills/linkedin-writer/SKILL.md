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
6. `_strip_wrapper` → `_finish` (only if capped or dangling colon) → `_enforce_length`
   → `apply_hashtags` (only when an audience is given) → `validator.validate_post`.
7. One-shot mode: at most **one** repair call, listing only the checks that failed.
   Refine mode (`services/refine_loop.refine`, `POST /api/writer/refine`, Writer
   "Refine loop" button): no blanket repair; one failing check per micro-iteration,
   free deterministic fixers first, model fixes rewrite **one sentence** and are
   capped (`max_llm_fixes`, default 3), anything not strictly better is rolled back.

## Hashtags

Built by `nlp/hashtags.audience_hashtags` from the audience text (word pairs and
content words) plus the author's own historical tags that share an audience word —
never left to the model. Every tag must trace to an audience word. No audience →
no generated tags, and that path behaves as before. Short audiences give 2–3 tags
on purpose; padding with single words (`#Family` for "family offices") is worse.

## Complete endings

The token cap sits just above the target length, so a long-running model is cut
mid-sentence. `LLM.generate` already trims capped replies (`harness/completion.py`),
but for posts `_finish` is authoritative: a colon at line end is **not** an ending,
a runtime-added "…" is a cut, and a trailing hashtag line is kept. A capped reply
with no complete sentence left sets `unfinished=True` → `complete_ending` fails.
Do not add a generic "must end with punctuation" check — 56 of the author's 77 posts
end on a URL, emoji or hashtags.

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
| Generic filler ("game-changer", "delve into") | **critical** banned-phrase check → drop the sentence if the post stays above the word floor, else rewrite that one sentence |
| Stops on "...these challenges:" | **critical** `complete_ending` → `_finish` |
| Copied sentence from an example | overlap check ≥ 0.6 → regenerate |

Before adding a phrase to `GENERIC_PHRASES`, count it in the author's 77 posts. Only
phrases with 0 uses were added; "game-changing", "streamline", "robust", "landscape"
are the author's own and stay allowed.

**Known blind spots (backtest `a3edbfd371`, 5/5 SUCCESS but not all publishable):**
topic drift into a retrieved example, example content restated as fact ("AI Business
Accountant, designed by our team"), biography pulled from old posts ("IIT Madras"),
and generic register ("in the realm of"). No check sees these yet — a refine loop
that passes the validator is not proof of a good post. Read the text.

## Model note

Measured on this machine: the 1.8B needs all of the above scaffolding and still
runs long. Set `JLA_MODEL_PREFER=large` for the 7B when quality matters more than
the ~3× latency.
