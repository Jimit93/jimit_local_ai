"""Classify user feedback to the layer that can actually fix it.

"Make it sound more like me" is not a prompt instruction. It is a symptom, and the
cause sits in one specific layer - the retrieved examples, the style profile, the
plan, the validator, or (last) the prompt. Appending the sentence to the next
prompt hides the cause and makes the model carry work a cheaper layer should do.

Classification is deterministic: regex over a small phrase table, no model call.
Each rule names the layer, the component to inspect, and whether the fix needs the
LLM at all. Feedback is persisted to `agent_memory` so it survives the session and
can be turned into a regression test.
"""
from __future__ import annotations

import re
import time

from backend.common import ProcessorResult, timed
from backend.db import recall, remember

# layer, pattern, component that owns the fix, action, whether the LLM is involved
_RULES: list[tuple[str, re.Pattern, str, str, bool]] = [
    ("VALIDATION", re.compile(
        r"\b(?:ai )?(?:jargon|buzzwords?|corporate speak|marketing speak|fluff|filler|"
        r"hype|salesy|cringe)\b", re.I),
     "nlp.features.GENERIC_PHRASES + validator.no_generic_phrases",
     "Add the offending phrases to GENERIC_PHRASES so the deterministic validator "
     "rejects them. Never spend an LLM call detecting a fixed phrase list.", False),

    ("VALIDATION", re.compile(
        r"\b(?:not (?:technical|specific) enough|too (?:shallow|vague|high[- ]level)|"
        r"needs? more (?:detail|specifics|depth))\b", re.I),
     "style_constraints.min_technical_density + validator.technical_density",
     "Raise the technical-density floor for this request; the validator already "
     "measures density and can fail the post before it is returned.", False),

    ("VALIDATION", re.compile(
        r"\b(?:no (?:proof|evidence|numbers)|where(?:'s| are) the (?:numbers|data)|"
        r"unsupported|ignored (?:my|the) (?:facts|numbers)|didn'?t use (?:my|the) "
        r"(?:facts|numbers))\b", re.I),
     "validator.facts_used",
     "The facts_used gate covers this. Check that facts were supplied and that the "
     "gate is in CRITICAL_CHECKS so the repair pass runs.", False),

    ("HARNESS", re.compile(
        r"\b(?:too long|too short|wrong length|shorten|trim it|too wordy|"
        r"too brief)\b", re.I),
     "linkedin_writer._target_words + LENGTH_PRESETS",
     "Length is a deterministic budget, not a prompt negotiation. Adjust the length "
     "preset or target_words; _enforce_length already trims by whole paragraphs.", False),

    ("HARNESS", re.compile(
        r"\b(?:too many|too few|no) hashtags?\b|\bhashtags? (?:are )?wrong\b", re.I),
     "linkedin_writer.build_plan hashtags + style_constraints.hashtags",
     "The plan takes the corpus p75. Respect the measured usage rate instead of the "
     "upper quartile.", False),

    ("HARNESS", re.compile(
        r"\b(?:cta|call to action) (?:is )?(?:wrong|off|bad)\b|\bwrong (?:cta|ending|"
        r"sign[- ]?off)\b|\b(?:i'?m|i am) not (?:hiring|recruiting)\b", re.I),
     "linkedin_writer.build_plan cta + style_engine.CTA_GUIDANCE",
     "Check cta_expected before forcing a CTA, and translate the detector label "
     "through CTA_GUIDANCE - never pass the raw label to the model.", False),

    ("CLASSIFICATION", re.compile(
        r"\bwrong audience\b|\b(?:aimed|written) at the wrong\b|\bnot for (?:this|that) "
        r"audience\b|\bwrong (?:pillar|topic|category)\b", re.I),
     "harness.task_classifier + ml.topics",
     "Audience and pillar are inputs and classifier outputs, not something the model "
     "should infer. Fix the rule or retrain the classifier; do not prompt around it.", False),

    ("RETRIEVAL", re.compile(
        r"\b(?:doesn'?t|does not) sound like me\b|\bnot my voice\b|\bsounds? like "
        r"(?:chatgpt|an ai|a robot|someone else)\b|\bmore of my (?:normal |usual )?"
        r"(?:writing |)style\b", re.I),
     "retrieval.index.search + linkedin_writer.select_examples",
     "Voice comes from the retrieved examples. Check which documents were selected "
     "and whether select_examples used Hit.doc_text (the whole post) rather than the "
     "matched chunk. Only touch the prompt if the right examples were retrieved and "
     "still ignored.", False),

    ("STYLE", re.compile(
        r"\b(?:hook|opening|first line) (?:is )?(?:weak|boring|bland|flat|generic)\b|"
        r"\bweak (?:hook|opening)\b", re.I),
     "style_engine.HOOK_GUIDANCE + build_plan.hook_type + validator",
     "The plan already picks a hook type from the measured distribution but nothing "
     "verifies the generated hook matches it. Add the check before touching the prompt.",
     False),

    ("STYLE", re.compile(
        r"\btoo generic\b|\bcould be anyone\b|\bbland\b|\bboring\b|\bforgettable\b", re.I),
     "select_examples + validator.no_generic_phrases + technical density",
     "Usually two causes at once: weak retrieved examples and no concrete evidence in "
     "the body. Check retrieval and facts_used before assuming the prompt is at fault.",
     False),

    ("MODEL", re.compile(
        r"\b(?:repeats? itself|loops?|garbled|incoherent|makes? no sense|"
        r"cut off mid)\b", re.I),
     "llm.manager + config.LLMSettings",
     "A decoding or context problem, not a style one. Check max_tokens, n_ctx and "
     "temperature before changing any instruction.", True),
]

_FALLBACK = ("PROMPT", "harness.prompt_builder",
             "No deterministic rule matched. Inspect the trace end to end before "
             "editing the prompt - the prompt is the last layer to change, not the "
             "first.", True)


def classify_feedback(text: str) -> ProcessorResult:
    """Map one piece of feedback to the layer that owns the fix. No LLM."""
    with timed() as elapsed:
        t = (text or "").strip()
        if not t:
            return ProcessorResult(None, 0.0, "feedback_classifier", elapsed(), "nlp",
                                   error="empty feedback")
        matches = []
        for layer, rx, component, action, needs_llm in _RULES:
            m = rx.search(t)
            if m:
                matches.append({
                    "layer": layer, "component": component, "action": action,
                    "needs_llm": needs_llm, "matched": m.group(0),
                })
        if not matches:
            layer, component, action, needs_llm = _FALLBACK
            matches = [{"layer": layer, "component": component, "action": action,
                        "needs_llm": needs_llm, "matched": None}]
            conf = 0.3
        else:
            # Several rules can legitimately fire ("too generic and too long").
            conf = 0.95 if len(matches) == 1 else 0.85
        return ProcessorResult(
            result={"feedback": t, "classifications": matches,
                    "primary_layer": matches[0]["layer"],
                    "deterministic_fix_available": not all(m["needs_llm"] for m in matches)},
            confidence=conf, processor="feedback_classifier(rules)",
            latency_ms=elapsed(), layer="nlp",
            meta={"answer": f"{matches[0]['layer']}: {matches[0]['component']}"},
        )


def record_feedback(text: str, post: str = "", run_id: str = "") -> dict:
    """Classify and persist. The stored record is the seed for a regression test."""
    res = classify_feedback(text)
    if not res.ok:
        return {"error": res.error}
    key = f"fb_{int(time.time())}"
    payload = dict(res.result)
    payload["post_excerpt"] = (post or "")[:400]
    payload["run_id"] = run_id
    remember("feedback", key, payload, source="user", confidence=res.confidence)
    return {"key": key, "confidence": res.confidence, **payload}


def list_feedback(limit: int = 50) -> list[dict]:
    rows = recall("feedback")
    return rows[:limit]
