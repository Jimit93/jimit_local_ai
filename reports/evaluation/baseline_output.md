# Baseline output — iteration 0

Unmodified system at checkpoint `7af8800`. Generated through the real frontend
(LinkedIn Writer → Generate), not by calling the service directly.

- **Model**: qwen2.5-7b-instruct-q4_0 via llama.cpp, CPU only
- **Captured**: `POST /api/writer/generate`, network request `10312.74`

## Brief as entered in the UI

| Field | Value |
|---|---|
| Topic | How I built a fully local AI agent that runs on a CPU-only laptop: a routing harness that sends every task to the cheapest processor that can answer it reliably, so the language model is the last resort instead of the default. |
| Audience | AI engineering hiring managers and companies that want AI automation built |
| Content pillar | engineering depth, proof |
| Length | Medium |
| CTA | Auto (your usual) |
| Facts | 18 measured numbers from benchmark run `b79f795399` (LLM calls −82.6%, tokens −73.5%, latency −83.0%, CPU −81.5%, accuracy 39.1% → 100%, 19/23 tasks with no LLM call) |

## Generated post

> I built a routing harness that sends every task to the cheapest processor that
> can answer it reliably, making the language model the last resort. This setup
> reduces LLM calls and latency, optimizing performance on a CPU-only laptop. 🛠️
>
> Try it out and see the difference in efficiency and accuracy. Hiring for more
> engineers with a practical approach to AI.

## What the system reported

| Metric | Value |
|---|---|
| Style match | **100%** |
| Status | SUCCESS |
| Validator issues | none |
| Words | 60 |
| Paragraphs | 2 |
| Hashtags | 0 |
| Emoji | 2 |
| Technical density | 0.1833 |
| Hook type (actual) | announcement (**planned: claim**) |
| CTA types detected | link, hiring |
| Sources retrieved | 2 (doc 5809 score 0.6851, doc 5807 score 0.6021) |
| LLM calls | 1 |
| Tokens | 956 (884 prompt / 72 completion) |
| Latency | 45.3 s |

## What is actually wrong with it

The system reported a perfect score. Two defects it could not see:

1. **Every supplied fact was ignored.** 18 measured numbers were passed in and the
   post contains none of them. For a post whose stated pillar is "proof", aimed at
   hiring managers, the evidence is the entire point. `numbers_grounded` passed
   *vacuously* — it only checks that numbers used are permitted, so a post with no
   numbers satisfies it trivially.

2. **"Hiring for more engineers" inverts the author's intent.** The audience is
   hiring managers who might hire Jimit; the post announces that Jimit is
   recruiting. This came from the plan forcing CTA `hiring` and the prompt passing
   that internal detector label to the model verbatim.

Neither defect is a model failure. Prompt reconstruction (byte-exact at 3,311
chars) confirms the model received the facts and the misleading CTA instruction.
