"""Context budgeting.

The local model runs at 5-15 tok/s, so every token in the prompt is paid for twice:
once to build it and once to process it. This module decides what is worth sending.
"""
from __future__ import annotations

import re

from backend.config import SETTINGS

# Rough but stable: llama tokenizers average ~3.6-4.0 chars/token on English prose.
CHARS_PER_TOKEN = 3.8


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text or "") / CHARS_PER_TOKEN))


def compress(text: str, max_chars: int) -> str:
    """Trim a block to a budget, cutting at a sentence/line boundary when possible."""
    t = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(t) <= max_chars:
        return t
    cut = t[:max_chars]
    for sep in ("\n\n", ". ", "\n", " "):
        i = cut.rfind(sep)
        if i > max_chars * 0.6:
            return cut[:i].rstrip() + " …"
    return cut.rstrip() + " …"


def dedupe_snippets(snippets: list[str], min_overlap: float = 0.8) -> list[str]:
    """Drop near-identical retrieved passages (the corpus has reposts)."""
    from backend.nlp.keywords import lexical_overlap

    kept: list[str] = []
    for s in snippets:
        if any(lexical_overlap(s, k) >= min_overlap for k in kept):
            continue
        kept.append(s)
    return kept


def build_context(
    blocks: list[tuple[str, str]],
    max_chars: int | None = None,
) -> tuple[str, dict]:
    """Assemble labelled blocks under a total character budget.

    `blocks` are (label, content) in priority order - earlier blocks get the space.
    Returns the rendered context and an accounting dict for the UI.
    """
    budget = max_chars or SETTINGS.harness.max_context_chars
    parts: list[str] = []
    used = 0
    included: list[str] = []
    dropped: list[str] = []
    for label, content in blocks:
        content = (content or "").strip()
        if not content:
            continue
        remaining = budget - used
        if remaining < 120:
            dropped.append(label)
            continue
        chunk = compress(content, remaining - len(label) - 6)
        parts.append(f"## {label}\n{chunk}")
        used += len(chunk) + len(label) + 6
        included.append(label)
    text = "\n\n".join(parts)
    return text, {
        "context_chars": len(text),
        "context_tokens_est": estimate_tokens(text),
        "blocks_included": included,
        "blocks_dropped": dropped,
        "budget_chars": budget,
    }


def extract_payload(task: str) -> str | None:
    """Pull the user-supplied text a task operates on.

    Supports: quoted text, a block after a colon, or an explicit --- fence. Returns
    None when the task carries no payload (the caller then works on the corpus).
    """
    t = (task or "").strip()
    fence = re.search(r"---\s*\n(.+?)(?:\n---|\Z)", t, re.S)
    if fence and len(fence.group(1).strip()) > 20:
        return fence.group(1).strip()
    quoted = re.search(r"[\"“']{1}([^\"”']{40,})[\"”']{1}", t, re.S)
    if quoted:
        return quoted.group(1).strip()
    if "\n" in t:
        head, rest = t.split("\n", 1)
        if len(rest.strip()) > 40 and len(head) < 160:
            return rest.strip()
    colon = re.search(r":\s*(.{60,})$", t, re.S)
    if colon:
        return colon.group(1).strip()
    return None
