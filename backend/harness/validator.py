"""Deterministic validation / final QC.

Nothing here calls a model: a generated post is checked against measured style
targets, and a numeric answer is checked by recomputing it.
"""
from __future__ import annotations

import re

from backend.nlp import features as F
from backend.nlp.keywords import lexical_overlap
from backend.tools.deterministic import calculate, text_stats

STATUS_OK = "SUCCESS"
STATUS_PARTIAL = "PARTIAL_SUCCESS"
STATUS_LOW = "LOW_CONFIDENCE"
STATUS_FAIL = "FAILURE"


def validate_numeric(question: str, answer: str) -> dict:
    """Recompute arithmetic independently and compare with the stated answer."""
    try:
        truth = calculate(question)["value"]
    except Exception:
        return {"checked": False, "status": STATUS_LOW, "reason": "not recomputable"}
    nums = [float(n.replace(",", "")) for n in re.findall(r"-?\d[\d,]*\.?\d*", answer or "")]
    ok = any(abs(n - float(truth)) < max(1e-6, abs(float(truth)) * 1e-9) for n in nums)
    return {
        "checked": True, "expected": truth, "found": nums[:5],
        "status": STATUS_OK if ok else STATUS_FAIL,
        "reason": "" if ok else "answer does not contain the computed value",
    }


# Checks that make a post unusable no matter how well it scores elsewhere.
CRITICAL_CHECKS = ("numbers_grounded", "no_placeholders", "not_copied")

_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
_MARKDOWN_RE = re.compile(r"\*\*|^#{1,6}\s|^\s*\|.*\|\s*$|__[^_]+__", re.M)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUM_RE.findall(text or "")}


def validate_post(
    text: str,
    constraints: dict,
    examples: list[str] | None = None,
    facts: str | None = None,
    topic: str | None = None,
) -> dict:
    """Score a generated post against the measured style profile.

    When `facts` are supplied they are the *only* permitted source of numbers: a
    model handed old posts as style examples will otherwise mine them for content
    and state last year's metrics as today's.
    """
    s = text_stats(text)
    feats = F.extract_features(text)
    issues: list[str] = []
    checks: dict[str, bool] = {}

    lo, hi = constraints.get("target_words", [40, 140])
    tolerance_lo, tolerance_hi = int(lo * 0.6), int(hi * 1.6)
    checks["length"] = tolerance_lo <= s["word_count"] <= tolerance_hi
    if not checks["length"]:
        issues.append(f"length {s['word_count']}w outside {tolerance_lo}-{tolerance_hi}w")

    max_sent = constraints.get("max_sentence_words", 40)
    checks["sentence_length"] = s["max_sentence_len"] <= max_sent * 1.5
    if not checks["sentence_length"]:
        issues.append(f"longest sentence {s['max_sentence_len']}w > {int(max_sent * 1.5)}w")

    h_lo, h_hi = constraints.get("hashtags", [0, 5])
    checks["hashtags"] = h_lo <= s["hashtag_count"] <= max(h_hi, 3)
    if not checks["hashtags"]:
        issues.append(f"{s['hashtag_count']} hashtags outside {h_lo}-{max(h_hi, 3)}")

    if not constraints.get("emoji_allowed", True):
        checks["emoji"] = s["emoji_count"] == 0
        if not checks["emoji"]:
            issues.append("emoji used but author rarely uses them")

    banned = [p for p in constraints.get("banned_phrases", F.GENERIC_PHRASES)
              if p in (text or "").lower()]
    checks["no_generic_phrases"] = not banned
    if banned:
        issues.append(f"generic phrases: {', '.join(banned[:3])}")

    min_td = constraints.get("min_technical_density", 0.0)
    checks["technical_density"] = feats["technical_density"] >= min_td * 0.5
    if not checks["technical_density"]:
        issues.append(
            f"technical density {feats['technical_density']} below author floor {min_td}"
        )

    # Small models leave template placeholders like "[Author's Name]" or "[Industry]".
    placeholders = re.findall(r"\[[^\]\n]{2,40}\]", text or "")
    checks["no_placeholders"] = not placeholders
    if placeholders:
        issues.append(f"unfilled placeholders: {', '.join(placeholders[:3])}")

    checks["no_meta_text"] = not re.search(
        r"^\s*(?:here(?:'s| is) (?:a|your)|sure[,!]|certainly|i'?ve (?:written|drafted)|"
        r"as an ai|note:)", (text or "").strip(), re.I
    )
    if not checks["no_meta_text"]:
        issues.append("output contains assistant preamble")

    checks["no_markdown"] = not _MARKDOWN_RE.search(text or "")
    if not checks["no_markdown"]:
        issues.append("markdown formatting (LinkedIn renders it literally)")

    if facts is not None and facts.strip():
        allowed = _numbers(facts) | _numbers(topic or "")
        used = _numbers(text)
        invented = sorted(used - allowed)
        checks["numbers_grounded"] = not invented
        if invented:
            issues.append(
                f"numbers not present in the supplied facts: {', '.join(invented[:4])}")

    copied = 0.0
    if examples:
        copied = max(lexical_overlap(text, e) for e in examples)
        checks["not_copied"] = copied < 0.6
        if not checks["not_copied"]:
            issues.append(f"too similar to a source example (overlap {copied})")

    passed = sum(1 for v in checks.values() if v)
    score = round(passed / max(1, len(checks)), 4)
    status = STATUS_OK if score >= 0.85 else (STATUS_PARTIAL if score >= 0.6 else STATUS_FAIL)

    # A fabricated number, an unfilled placeholder or a copied sentence is a hard
    # failure, however good the rest of the post looks. Without this a post that
    # invents a metric scores 0.9 and is returned as SUCCESS.
    failed_critical = [k for k in CRITICAL_CHECKS if checks.get(k) is False]
    if failed_critical:
        status = STATUS_FAIL if score < 0.85 else STATUS_PARTIAL
    return {
        "style_match": score,
        "status": status,
        "checks": checks,
        "issues": issues,
        "stats": {
            "words": s["word_count"], "sentences": s["sentence_count"],
            "paragraphs": s["paragraph_count"], "hashtags": s["hashtag_count"],
            "emoji": s["emoji_count"], "max_sentence_words": s["max_sentence_len"],
            "technical_density": feats["technical_density"],
            "hook_type": feats["hook_type"], "cta_types": feats["cta_types"],
            "readability": feats["readability"],
        },
        "max_example_overlap": round(copied, 4),
    }


def validate_generic(answer: str, min_chars: int = 2) -> dict:
    """Last-line QC for free-text answers."""
    a = (answer or "").strip()
    issues = []
    if len(a) < min_chars:
        issues.append("empty or trivial answer")
    if re.search(r"\b(?:as an ai language model|i cannot browse|i don't have access to)\b", a, re.I):
        issues.append("model refusal boilerplate")
    if re.search(r"(.{30,}?)\1{2,}", a):
        issues.append("repeated text loop")
    return {
        "status": STATUS_OK if not issues else STATUS_PARTIAL,
        "issues": issues,
        "chars": len(a),
    }
