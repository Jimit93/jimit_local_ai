"""Compact structured prompts.

Static system instructions are constants (cached by the runtime's prompt cache and
never re-sent as part of the user turn). The user turn carries only what changes.
"""
from __future__ import annotations

SYSTEM_BASE = (
    "You are a local AI assistant running offline on the user's own machine. "
    "Be accurate, concise and concrete. If the provided context does not contain "
    "the answer, say so instead of inventing facts."
)

SYSTEM_WRITER = (
    "You are a LinkedIn ghostwriter that reproduces one specific author's voice. "
    "Write only the post text - no preamble, no explanation, no markdown headers, "
    "no quotation marks around the post. Never invent metrics, dates, company names "
    "or achievements that are not in the provided context."
)

SYSTEM_ANALYST = (
    "You interpret pre-computed statistics. The numbers are already correct - do not "
    "recalculate them, do not contradict them, and do not add numbers of your own."
)


def build_prompt(
    *,
    objective: str,
    task: str,
    context: str = "",
    constraints: list[str] | None = None,
    output_format: str = "",
) -> str:
    """ROLE/OBJECTIVE/CONTEXT/CONSTRAINTS/TASK/OUTPUT - only the parts that exist."""
    parts = [f"# OBJECTIVE\n{objective}"]
    if context:
        parts.append(f"# CONTEXT\n{context}")
    if constraints:
        parts.append("# CONSTRAINTS\n" + "\n".join(f"- {c}" for c in constraints))
    parts.append(f"# TASK\n{task}")
    if output_format:
        parts.append(f"# OUTPUT\n{output_format}")
    return "\n\n".join(parts)


def writer_prompt(
    *,
    topic: str,
    style_rules: list[str],
    examples: list[str],
    facts: str = "",
    audience: str = "",
    pillar: str = "",
    cta: str = "",
    target_words: tuple[int, int] = (60, 120),
    constraints_extra: list[str] | None = None,
) -> str:
    """Style arrives as imperative rules, never as statistics.

    Anything that looks like data in the context gets reproduced as content by
    small models, so the measured profile is converted to instructions upstream.
    """
    ctx_parts = []
    if examples:
        ctx_parts.append(
            "EXAMPLES OF THE AUTHOR'S OWN POSTS - copy the voice and structure, "
            "never the sentences, and treat their claims as history, not as facts "
            "about today:\n" + "\n\n---\n".join(examples)
        )
    if facts:
        ctx_parts.append(
            "MEASURED FACTS - these are the evidence for this post. Quote the most "
            "relevant of these numbers exactly, and add no others:\n" + facts
        )

    cons = list(style_rules)
    cons += [
        "No generic LinkedIn filler and no hype words.",
        "Do not copy any sentence from the examples verbatim.",
        "Use only the facts given above; if a number is not provided, do not state one.",
    ]
    if facts:
        # Measured: the three instructions above are all restrictive, and with no
        # counterweight the model's safest move is to state no number at all - it
        # cited 0 of 18 supplied numbers twice in a row. One obligating instruction
        # is cheaper than the repair call that would otherwise be needed.
        cons.append("Include at least two of the measured numbers above, written "
                    "exactly as given, as the concrete evidence in the post.")
    if audience:
        cons.append(f"Audience: {audience}.")
    if pillar:
        cons.append(f"Content pillar: {pillar}.")
    if cta:
        # `cta` arrives as guidance text, not as an internal detector label.
        cons.append(f"Close the post with {cta}.")
    cons += constraints_extra or []
    return build_prompt(
        objective="Write one LinkedIn post in the author's established voice.",
        task=f"Topic: {topic}",
        context="\n\n".join(ctx_parts),
        constraints=cons,
        output_format=(
            f"The post text only, {target_words[0]}-{target_words[1]} words. "
            "No title, no headings, no labels such as 'Claim:' or 'Greeting:', "
            "no commentary about writing style, no word count."
        ),
    )
