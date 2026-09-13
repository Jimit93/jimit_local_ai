"""Parse model-proposed edits and apply them deterministically, or refuse.

A model proposes; this module decides. Every SEARCH part must match the current file
exactly once (with two tolerant fallbacks: trailing whitespace, then uniform
re-indentation). Edits to files outside the allowed set, to secret files, or containing
redaction markers are rejected, and so is a whole-file rewrite that deletes most of
the file. Application is atomic per file and goes through the checkpointed write tool.
"""
from __future__ import annotations

import difflib
import re
import textwrap
from dataclasses import asdict, dataclass, field

from backend.agent import code_tools as C
from backend.agent import policy

_BLOCK = re.compile(
    r"(?:^|\n)[ \t]*(?:[*#>`\-\s]*(?:File|Path|Filename)?:?\s*)?`?(?P<path>[\w./\\-]+\.[A-Za-z0-9]+)`?\**[ \t]*\n"
    r"(?:```[\w+-]*[ \t]*\n)?"
    r"[ \t]*<{5,9} ?SEARCH[ \t]*\n(?P<search>.*?)\n?[ \t]*={5,9}[ \t]*\n(?P<replace>.*?)\n?[ \t]*>{5,9} ?REPLACE",
    re.S)
_FENCE = re.compile(r"```[\w+-]*\n(.*?)```", re.S)


@dataclass
class EditBlock:
    path: str
    search: str
    replace: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ApplyReport:
    ok: bool
    applied: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    diff: str = ""
    syntax_errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_blocks(text: str) -> list[EditBlock]:
    blocks = []
    for m in _BLOCK.finditer(text or ""):
        # Strip only a literal "./" prefix; "../" must reach the policy and be refused.
        path = re.sub(r"^(?:\./)+", "", m.group("path").replace("\\", "/"))
        blocks.append(EditBlock(path, m.group("search"), m.group("replace")))
    return blocks


def _find_unique(content: str, search: str) -> tuple[int, int] | str:
    """Return (start, end) of the unique match, or an error string."""
    if not search.strip():
        return "empty SEARCH on an existing file"
    n = content.count(search)
    if n == 1:
        i = content.index(search)
        return i, i + len(search)
    if n > 1:
        return f"SEARCH text matches {n} times; it must be unique"
    # Fallback 1: ignore trailing whitespace per line.
    lines = content.splitlines(keepends=True)
    s_lines = [ln.rstrip() for ln in search.strip("\n").splitlines()]
    k = len(s_lines)
    hits = [i for i in range(len(lines) - k + 1) if [ln.rstrip() for ln in lines[i:i + k]] == s_lines]
    # Fallback 2: same code at a different uniform indentation.
    if not hits:
        ded = [ln.rstrip() for ln in textwrap.dedent("\n".join(s_lines)).splitlines()]
        hits = [i for i in range(len(lines) - k + 1)
                if [ln.rstrip() for ln in textwrap.dedent("".join(lines[i:i + k])).splitlines()] == ded]
    if len(hits) == 1:
        start = sum(len(ln) for ln in lines[:hits[0]])
        end = start + sum(len(ln) for ln in lines[hits[0]:hits[0] + k])
        return start, end
    if len(hits) > 1:
        return f"SEARCH text matches {len(hits)} times (ignoring whitespace); it must be unique"
    first = next((ln.strip() for ln in s_lines if ln.strip()), "")
    close = difflib.get_close_matches(first, [ln.strip() for ln in lines if ln.strip()], n=2, cutoff=0.6)
    return "SEARCH text not found in the file" + (f"; closest existing lines: {close}" if close else "")


def _reindent(replace: str, original: str, search: str) -> str:
    """If the match used the indentation fallback, shift REPLACE by the same offset."""
    def indent(s: str) -> int:
        for ln in s.splitlines():
            if ln.strip():
                return len(ln) - len(ln.lstrip())
        return 0

    delta = indent(original) - indent(search)
    if delta <= 0:
        return replace
    pad = " " * delta
    return "\n".join(pad + ln if ln.strip() else ln for ln in replace.splitlines()) + ("\n" if replace.endswith("\n") else "")


def apply_blocks(workspace: str, blocks: list[EditBlock], *, allowed_files: list[str] | None = None,
                 allow_new_files: bool = False, checkpoint_id: str | None = None, task_id: str | None = None,
                 dry_run: bool = False) -> ApplyReport:
    rep = ApplyReport(ok=False)
    if not blocks:
        rep.errors.append("no edit blocks found in the model output")
        return rep
    ws = policy.resolve_workspace(workspace)
    allowed = {a.replace("\\", "/").lower() for a in (allowed_files or [])}
    pending: dict[str, tuple[str, str]] = {}  # rel -> (original, new)
    for b in blocks:
        try:
            target = policy.resolve_path(ws, b.path, for_write=True)
        except policy.PolicyError as e:
            rep.errors.append(f"{b.path}: {e}")
            continue
        rel = target.relative_to(ws).as_posix()
        if policy.is_secret_file(target):
            rep.errors.append(f"{rel}: secret files are never edited")
            continue
        if "[REDACTED]" in b.search or "[REDACTED]" in b.replace:
            rep.errors.append(f"{rel}: edit touches redacted content; refusing")
            continue
        exists = target.is_file()
        if not exists:
            if not allow_new_files or b.search.strip():
                rep.errors.append(f"{rel}: file does not exist" + ("" if allow_new_files else " (new files not allowed for this task)"))
                continue
            orig, cur = "", pending.get(rel, ("", ""))[1]
            pending[rel] = (orig, cur + b.replace)
            continue
        if allowed and rel.lower() not in allowed:
            rep.errors.append(f"{rel}: not one of the files this task may edit ({', '.join(sorted(allowed))})")
            continue
        orig, cur = pending.get(rel) or (C._read_text(target)[0],) * 2
        crlf = "\r\n" in cur
        search, replace = b.search, b.replace
        if crlf:
            search, replace = search.replace("\r\n", "\n").replace("\n", "\r\n"), replace.replace("\r\n", "\n").replace("\n", "\r\n")
        found = _find_unique(cur, search)
        if isinstance(found, str):
            rep.errors.append(f"{rel}: {found}")
            continue
        s, e = found
        matched = cur[s:e]
        new_text = _reindent(replace, matched, search) if matched != search else replace
        if matched.endswith(("\n", "\r\n")) and new_text and not new_text.endswith("\n"):
            new_text += "\r\n" if crlf else "\n"
        pending[rel] = (orig, cur[:s] + new_text + cur[e:])
        rep.applied.append({"path": rel, "exact": matched == search, "removed_lines": matched.count("\n"),
                            "added_lines": new_text.count("\n")})
    if rep.errors:
        return rep
    diffs = []
    for rel, (orig, new) in pending.items():
        if orig and len(new.splitlines()) < 0.6 * len(orig.splitlines()) and len(orig.splitlines()) > 30:
            rep.errors.append(f"{rel}: edit would delete {len(orig.splitlines()) - len(new.splitlines())} of "
                              f"{len(orig.splitlines())} lines; refusing a destructive rewrite")
        diffs.append("".join(difflib.unified_diff(orig.splitlines(True), new.splitlines(True),
                                                  fromfile=f"a/{rel}", tofile=f"b/{rel}", n=2)))
        syn = C._quick_syntax(rel, new)
        if syn.get("ok") is False:
            rep.syntax_errors.append({"path": rel, **syn})
    rep.diff = policy.redact("\n".join(diffs))[0][:20000]
    if rep.errors:
        return rep
    if rep.syntax_errors:
        # Never write code that does not parse; the repair loop gets the exact error.
        rep.errors += [f"{e['path']}:{e.get('line')}: {e.get('error')}" for e in rep.syntax_errors]
        return rep
    if not dry_run:
        for rel, (_orig, new) in pending.items():
            C.write_file(str(ws), rel, new, checkpoint_id=checkpoint_id, task_id=task_id)
            rep.files_changed.append(rel)
    else:
        rep.files_changed = list(pending)
    rep.ok = True
    return rep


# ------------------------------------------------------ deterministic rename

def rename_symbol(workspace: str, old: str, new: str, *, files: list[str] | None = None,
                  checkpoint_id: str | None = None, task_id: str | None = None, dry_run: bool = False) -> ApplyReport:
    """Token-level rename in Python files (strings and comments untouched). No model."""
    import io
    import keyword
    import tokenize

    rep = ApplyReport(ok=False)
    if not (old.isidentifier() and new.isidentifier()) or keyword.iskeyword(new):
        rep.errors.append("both names must be valid identifiers")
        return rep
    ws = policy.resolve_workspace(workspace)
    targets = files or [rel for rel, _ in C.iter_files(ws) if rel.endswith(".py")]
    diffs = []
    for rel in targets:
        p = policy.resolve_path(ws, rel, for_write=True)
        try:
            src = C._read_text(p)[0]
        except (OSError, ValueError):
            continue
        if old not in src:
            continue
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
        except (tokenize.TokenError, SyntaxError) as e:
            rep.errors.append(f"{rel}: cannot tokenize ({e})")
            continue
        lines = src.splitlines(keepends=True)
        offsets = [0]
        for ln in lines:
            offsets.append(offsets[-1] + len(ln))
        spans = [(offsets[t.start[0] - 1] + t.start[1], offsets[t.end[0] - 1] + t.end[1])
                 for t in toks if t.type == tokenize.NAME and t.string == old]
        if not spans:
            continue
        out, last = [], 0
        for s, e in spans:
            out.append(src[last:s])
            out.append(new)
            last = e
        out.append(src[last:])
        new_src = "".join(out)
        diffs.append("".join(difflib.unified_diff(src.splitlines(True), new_src.splitlines(True),
                                                  fromfile=f"a/{rel}", tofile=f"b/{rel}", n=1)))
        rep.applied.append({"path": rel, "occurrences": len(spans)})
        if not dry_run:
            C.write_file(str(ws), rel, new_src, checkpoint_id=checkpoint_id, task_id=task_id)
        rep.files_changed.append(rel)
    rep.diff = "\n".join(diffs)[:20000]
    rep.ok = not rep.errors and bool(rep.files_changed)
    if not rep.files_changed and not rep.errors:
        rep.errors.append(f"no Python token named '{old}' found")
    return rep
