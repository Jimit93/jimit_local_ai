"""Make model output end cleanly inside its token limit. Deterministic; no extra model call.

Two layers:
  1. prevention - `word_budget(max_tokens)` gives prompts an explicit length to plan for;
  2. repair     - when the runtime reports the generation stopped at the limit
                  (stop_reason "length"), the unfinished tail is removed:
                    prose -> cut back to the last complete sentence / list item,
                            an unclosed code fence is closed or dropped;
                    edits -> an unterminated SEARCH/REPLACE block is dropped
                            (applying half a block would corrupt the file).

Output that finished naturally is returned unchanged.
"""
from __future__ import annotations

import re

WORDS_PER_TOKEN = 0.72  # measured tokenizers average ~1.35-1.4 tokens per English word
TRUNCATED_REASONS = {"length", "max_tokens", "max_length"}

# A colon is not an ending: after truncation it introduces content that is now missing.
_SENT_END = re.compile(r"""[.!?…](?:["')\]]+)?(?=\s|$)""")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.M)


def word_budget(max_tokens: int, reserve: float = 0.85) -> int:
    """Words the model should aim for so the answer finishes before the hard limit."""
    return max(8, int(max_tokens * WORDS_PER_TOKEN * reserve))


def budget_instruction(max_tokens: int) -> str:
    return (f"Keep the whole reply under {word_budget(max_tokens)} words and finish with a "
            "complete sentence; if more detail exists, summarise it instead of starting it.")


def is_truncated(stop_reason: str | None) -> bool:
    return (stop_reason or "").lower() in TRUNCATED_REASONS


def _close_or_drop_fence(text: str) -> tuple[str, str | None]:
    if text.count("```") % 2 == 0:
        return text, None
    start = text.rfind("```")
    body = text[start:]
    # A fence with at least one complete line of code is worth keeping, closed.
    lines = body.split("\n")
    if len(lines) > 2:
        kept = "\n".join(lines[:-1]).rstrip()
        return text[:start] + kept + "\n```", "closed_code_fence"
    return text[:start].rstrip(), "dropped_partial_code_fence"


_HEADING = re.compile(r"(?:\n|^)[ \t]*(?:#{1,6}[ \t]+[^\n]*|\*\*[^\n*]+\*\*:?)[ \t]*$")


def _drop_trailing_headings(t: str) -> str:
    """A heading (or bold label) with nothing under it reads as unfinished."""
    prev = None
    while prev != t:
        prev = t
        t = _HEADING.sub("", t).rstrip()
    return t


_LEADIN = re.compile(r"(?:\n|^)[^\n]*:\s*$")


def _drop_dangling_leadins(t: str) -> str:
    """'Here are the reasons:' / '1. **Step**:' with nothing after them read as cut off."""
    prev = None
    while prev != t and t:
        prev = t
        t = _drop_trailing_headings(t)
        if t.endswith(":"):
            line_start = t.rfind("\n") + 1
            line = t[line_start:]
            ends = [m.end() for m in _SENT_END.finditer(line)
                    if not re.fullmatch(r"\s*\d+[.)]", line[:m.end()])]
            # Keep complete sentences on the same line before the lead-in, drop the rest.
            t = (t[:line_start] + line[:ends[-1]]).rstrip() if ends else t[:line_start].rstrip()
    return t


def trim_prose(text: str) -> tuple[str, str]:
    """Cut an unfinished tail back to the last complete sentence or list item."""
    t, action = _trim_prose(text)
    if action not in ("already_complete", "closed_code_fence", "empty"):
        cleaned = _drop_dangling_leadins(t)
        if cleaned and cleaned != t:
            return cleaned, action + "+dropped_dangling_leadin"
    return t, action


def _trim_prose(text: str) -> tuple[str, str]:
    t = (text or "").rstrip()
    t, fence_action = _close_or_drop_fence(t)
    if fence_action == "closed_code_fence":
        return t, fence_action
    if not t:
        return t, "empty"
    if re.search(r"""[.!?…](?:["')\]]+)?$|```$""", t):
        return t, fence_action or "already_complete"
    # "3." at the start of a line is a list marker, not the end of a sentence.
    ends = [m.end() for m in _SENT_END.finditer(t)
            if not re.fullmatch(r"\s*\d+[.)]", t[t.rfind("\n", 0, m.start()) + 1:m.end()])]
    last_sentence = ends[-1] if ends else 0
    # A final list item that is itself complete (the list line before the cut) also counts.
    lines = t.split("\n")
    if len(lines) > 1 and _LIST_ITEM.match(lines[-1] or ""):
        list_cut = len("\n".join(lines[:-1]))
        if list_cut > last_sentence:
            return t[:list_cut].rstrip(), "dropped_partial_list_item"
    if last_sentence >= max(20, int(len(t) * 0.25)):
        return t[:last_sentence].rstrip(), "trimmed_to_last_sentence"
    # Almost nothing complete: keep the text but make it read as deliberately cut.
    cut = t.rsplit(" ", 1)[0] if " " in t else t
    return cut.rstrip(",;:-– ") + "…", "ellipsis_no_complete_sentence"


_BLOCK_END = re.compile(r">{5,9} ?REPLACE")
_BLOCK_START = re.compile(r"<{5,9} ?SEARCH")


def trim_edits(text: str) -> tuple[str, str]:
    t = text or ""
    starts = [m.start() for m in _BLOCK_START.finditer(t)]
    ends = [m.end() for m in _BLOCK_END.finditer(t)]
    if not starts or (ends and ends[-1] > starts[-1]):
        return t, "already_complete"
    last_end = ends[-1] if ends else 0
    return t[:last_end].rstrip(), "dropped_unterminated_edit_block"


def finish(text: str, stop_reason: str | None, mode: str = "prose") -> tuple[str, dict]:
    """Return (text, meta). Only acts on output that actually hit the token limit."""
    meta = {"truncated": is_truncated(stop_reason), "action": None, "trimmed_chars": 0}
    if not meta["truncated"] or mode == "none":
        return text, meta
    new, action = trim_edits(text) if mode == "edits" else trim_prose(text)
    meta.update(action=action, trimmed_chars=max(0, len(text or "") - len(new)))
    return new, meta
