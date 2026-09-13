"""Deterministic tools. No LLM anywhere in this module.

Every function is pure-Python, exact, and fast enough that calling a 5 tok/s local
model for the same job would be absurd.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import operator as op
import os
import re
import statistics as stats
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------- math

_BINOPS: dict[type, Callable] = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow,
}
_UNARY: dict[type, Callable] = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS: dict[str, Callable] = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "sum": lambda *a: sum(a[0]) if len(a) == 1 and isinstance(a[0], (list, tuple)) else sum(a),
    "log": math.log, "log10": math.log10, "exp": math.exp, "floor": math.floor,
    "ceil": math.ceil, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pow": pow, "factorial": math.factorial,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# Percentage forms that appear in ordinary questions.
_PCT_OF = re.compile(r"(-?[\d,.]+)\s*%\s*of\s*(-?[\d,.]+)", re.I)
_PCT_INC = re.compile(r"(-?[\d,.]+)\s*(?:\+|plus|increased by)\s*(-?[\d,.]+)\s*%", re.I)
_PCT_DEC = re.compile(r"(-?[\d,.]+)\s*(?:-|minus|decreased by)\s*(-?[\d,.]+)\s*%", re.I)

_WORD_OPS = [
    (re.compile(r"\b(?:multiplied by|times)\b", re.I), "*"),
    (re.compile(r"\b(?:divided by|over)\b", re.I), "/"),
    (re.compile(r"\b(?:plus|added to)\b", re.I), "+"),
    (re.compile(r"\b(?:minus|less)\b", re.I), "-"),
    (re.compile(r"\bto the power of\b", re.I), "**"),
    (re.compile(r"\bsquared\b", re.I), "**2"),
    (re.compile(r"\bx\b"), "*"),
]
_STRIP_PREFIX = re.compile(
    r"^\s*(?:what\s+is|whats|what's|calculate|compute|evaluate|how much is|solve|=)\s*", re.I
)
# Typographic operators people paste ("345 × 67", "10 ÷ 4", "5 − 3").
_UNICODE_OPS = str.maketrans({"×": "*", "✕": "*", "÷": "/", "−": "-"})


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"unknown function {node.func.id}")
        return fn(*[_eval_node(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]  # type: ignore[return-value]
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Safe arithmetic via AST. Returns {'value': float, 'expression': normalised}."""
    expr = _STRIP_PREFIX.sub("", str(expression).translate(_UNICODE_OPS)).strip().rstrip("?=.").strip()

    m = _PCT_OF.search(expr)
    if m:
        pct, base = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        return {"value": pct / 100.0 * base, "expression": f"{pct}% of {base}"}
    m = _PCT_INC.search(expr)
    if m:
        base, pct = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        return {"value": base * (1 + pct / 100.0), "expression": f"{base} +{pct}%"}
    m = _PCT_DEC.search(expr)
    if m:
        base, pct = float(m.group(1).replace(",", "")), float(m.group(2).replace(",", ""))
        return {"value": base * (1 - pct / 100.0), "expression": f"{base} -{pct}%"}

    norm = expr
    for rx, rep in _WORD_OPS:
        norm = rx.sub(rep, norm)
    # Drop thousands separators only (keep commas that separate function arguments).
    norm = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", norm)
    norm = norm.replace("^", "**")
    # `%` survives as the modulo operator; percentage phrasings were handled above.
    norm = re.sub(r"[^0-9+\-*/%(),. a-zA-Z_]", "", norm)
    value = _eval_node(ast.parse(norm, mode="eval"))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {"value": value, "expression": norm.strip()}


def looks_like_math(text: str) -> bool:
    """Cheap gate so the router can try the calculator without paying for a parse."""
    t = _STRIP_PREFIX.sub("", text.strip().translate(_UNICODE_OPS)).rstrip("?")
    if not re.search(r"\d", t):
        return False
    if _PCT_OF.search(t) or _PCT_INC.search(t) or _PCT_DEC.search(t):
        return True
    if len(t) > 120:
        return False
    stripped = re.sub(r"\b(?:multiplied by|divided by|times|plus|minus|over|squared|to the power of|x)\b",
                      "*", t, flags=re.I)
    return bool(re.fullmatch(r"[\s\d+\-*/^().,%]+", stripped)) and bool(
        re.search(r"[+\-*/^%]", stripped)
    )


# ----------------------------------------------------------------------- datetime

_DATE_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
              "%d %b %Y", "%d %B %Y", "%Y-%m-%dT%H:%M:%S")


def parse_date(text: str) -> datetime | None:
    text = text.strip()
    for f in _DATE_FMTS:
        try:
            return datetime.strptime(text, f)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def now_info(tz_offset_hours: float = 0.0) -> dict:
    d = datetime.now(timezone.utc) + timedelta(hours=tz_offset_hours)
    return {
        "iso": d.isoformat(timespec="seconds"),
        "date": d.date().isoformat(),
        "time": d.strftime("%H:%M:%S"),
        "weekday": d.strftime("%A"),
        "week": int(d.strftime("%V")),
        "unix": int(d.timestamp()),
    }


def date_diff(a: str, b: str) -> dict:
    da, db = parse_date(a), parse_date(b)
    if not da or not db:
        raise ValueError("unparseable date")
    delta = db - da
    return {"days": delta.days, "seconds": int(delta.total_seconds()), "weeks": delta.days // 7}


def date_add(a: str, days: int = 0, hours: int = 0) -> dict:
    da = parse_date(a)
    if not da:
        raise ValueError("unparseable date")
    d = da + timedelta(days=days, hours=hours)
    return {"iso": d.isoformat(timespec="seconds"), "weekday": d.strftime("%A")}


# ---------------------------------------------------------------------- text ops

EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF" "\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF" "\U00002B00-\U00002BFF" "\U0000FE0F" "]"
)
URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.I)
HASHTAG_RE = re.compile(r"(?<!\w)#(\w{2,50})")
MENTION_RE = re.compile(r"(?<!\w)@([\w.\-]{2,60})")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\s ]+(?=[\"'(\[]?[A-Z0-9#])|\n+")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
BULLET_RE = re.compile(r"^\s*(?:[-*•–▪✅➡→➜]|\d+[.)])\s+", re.M)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def char_count(text: str, include_spaces: bool = True) -> int:
    t = text or ""
    return len(t) if include_spaces else len(re.sub(r"\s", "", t))


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in SENT_SPLIT_RE.split(text or "") if s and s.strip()]
    return parts


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text or "") if p.strip()]


def extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text or "")


def extract_hashtags(text: str) -> list[str]:
    return ["#" + h for h in HASHTAG_RE.findall(text or "")]


def extract_mentions(text: str) -> list[str]:
    return MENTION_RE.findall(text or "")


def count_emoji(text: str) -> int:
    return len(EMOJI_RE.findall(text or ""))


def text_stats(text: str) -> dict:
    """One pass over the text producing every deterministic surface statistic."""
    t = text or ""
    words = WORD_RE.findall(t)
    sents = sentences(t)
    paras = paragraphs(t)
    slens = [len(WORD_RE.findall(s)) for s in sents] or [0]
    return {
        "char_count": len(t),
        "char_count_no_spaces": len(re.sub(r"\s", "", t)),
        "word_count": len(words),
        "unique_words": len({w.lower() for w in words}),
        "sentence_count": len(sents),
        "paragraph_count": len(paras),
        "avg_sentence_len": round(sum(slens) / len(slens), 2),
        "max_sentence_len": max(slens),
        "min_sentence_len": min(slens),
        "avg_word_len": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
        "hashtags": extract_hashtags(t),
        "hashtag_count": len(extract_hashtags(t)),
        "urls": extract_urls(t),
        "url_count": len(extract_urls(t)),
        "mention_count": len(extract_mentions(t)),
        "emoji_count": count_emoji(t),
        "question_count": t.count("?"),
        "exclamation_count": t.count("!"),
        "number_count": len(re.findall(r"(?<![\w.])\d+(?:[.,]\d+)?", t)),
        "bullet_count": len(BULLET_RE.findall(t)),
        "caps_word_count": len([w for w in words if len(w) > 2 and w.isupper()]),
        "colon_count": t.count(":"),
        "dash_count": len(re.findall(r"[—–]|--", t)),
        "line_count": len([ln for ln in t.splitlines() if ln.strip()]),
        "type_token_ratio": round(len({w.lower() for w in words}) / len(words), 4) if words else 0.0,
    }


def regex_search(pattern: str, texts: Iterable[str], flags: int = re.I) -> list[dict]:
    rx = re.compile(pattern, flags)
    out = []
    for i, t in enumerate(texts):
        ms = [m.group(0) for m in rx.finditer(t or "")]
        if ms:
            out.append({"index": i, "matches": ms, "count": len(ms)})
    return out


def keyword_filter(
    records: list[dict], field: str, keywords: list[str], mode: str = "any"
) -> list[dict]:
    """Exact case-insensitive substring filter. No embeddings, no model."""
    kws = [k.lower() for k in keywords]
    out = []
    for r in records:
        hay = str(r.get(field, "")).lower()
        hits = [k for k in kws if k in hay]
        if (mode == "all" and len(hits) == len(kws)) or (mode == "any" and hits):
            out.append({**r, "_matched": hits})
    return out


def dedupe(items: list[str], normalize: bool = True) -> dict:
    """Exact duplicate detection by normalised hash."""
    seen: dict[str, int] = {}
    dups: list[dict] = []
    unique: list[str] = []
    for i, s in enumerate(items):
        key = re.sub(r"\s+", " ", (s or "").strip().lower()) if normalize else (s or "")
        h = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()
        if h in seen:
            dups.append({"index": i, "duplicate_of": seen[h]})
        else:
            seen[h] = i
            unique.append(s)
    return {"unique_count": len(unique), "duplicate_count": len(dups), "duplicates": dups,
            "unique": unique}


# -------------------------------------------------------------- structured data

def parse_json(text: str) -> Any:
    return json.loads(text)


def parse_csv(text: str, delimiter: str = ",") -> list[dict]:
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def read_csv_file(path: str, limit: int | None = None, encoding: str = "utf-8-sig") -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding=encoding, errors="replace", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if limit is not None and i >= limit:
                break
            rows.append(row)
    return rows


def sort_records(records: list[dict], key: str, reverse: bool = False) -> list[dict]:
    def _k(r):
        v = r.get(key)
        try:
            return (0, float(v))
        except (TypeError, ValueError):
            return (1, str(v).lower())

    return sorted(records, key=_k, reverse=reverse)


def filter_records(records: list[dict], field: str, opname: str, value: Any) -> list[dict]:
    ops = {
        "eq": lambda a, b: str(a) == str(b), "ne": lambda a, b: str(a) != str(b),
        "gt": lambda a, b: float(a) > float(b), "lt": lambda a, b: float(a) < float(b),
        "gte": lambda a, b: float(a) >= float(b), "lte": lambda a, b: float(a) <= float(b),
        "contains": lambda a, b: str(b).lower() in str(a).lower(),
        "startswith": lambda a, b: str(a).lower().startswith(str(b).lower()),
    }
    fn = ops[opname]
    out = []
    for r in records:
        try:
            if fn(r.get(field), value):
                out.append(r)
        except (TypeError, ValueError):
            continue
    return out


def aggregate(records: list[dict], field: str, how: str = "sum", group_by: str | None = None):
    def _vals(rs):
        vs = []
        for r in rs:
            try:
                vs.append(float(r.get(field)))
            except (TypeError, ValueError):
                continue
        return vs

    def _apply(vs):
        if how == "count":
            return len(vs)
        if not vs:
            return None
        return {"sum": sum, "mean": stats.fmean, "min": min, "max": max,
                "median": stats.median}[how](vs)

    if group_by:
        groups: dict[str, list[dict]] = {}
        for r in records:
            groups.setdefault(str(r.get(group_by, "")), []).append(r)
        return {k: _apply(_vals(v)) if how != "count" else len(v) for k, v in groups.items()}
    return _apply(_vals(records)) if how != "count" else len(records)


def describe(numbers: list[float]) -> dict:
    xs = [float(x) for x in numbers]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs), "sum": sum(xs), "mean": round(stats.fmean(xs), 6),
        "median": stats.median(xs), "min": min(xs), "max": max(xs),
        "stdev": round(stats.pstdev(xs), 6) if len(xs) > 1 else 0.0,
        "p25": round(sorted(xs)[max(0, int(len(xs) * 0.25) - 0)], 6),
        "p75": round(sorted(xs)[min(len(xs) - 1, int(len(xs) * 0.75))], 6),
    }


# ------------------------------------------------------------------- filesystem

def file_metadata(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    st = p.stat()
    return {
        "path": str(p), "name": p.name, "suffix": p.suffix, "is_dir": p.is_dir(),
        "size_bytes": st.st_size, "size_mb": round(st.st_size / 1e6, 3),
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    }


def list_dir(path: str, pattern: str = "*", limit: int = 500) -> list[dict]:
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(path)
    out = []
    for i, c in enumerate(sorted(p.glob(pattern))):
        if i >= limit:
            break
        try:
            out.append({"name": c.name, "is_dir": c.is_dir(), "size_bytes": c.stat().st_size})
        except OSError:
            continue
    return out


# ------------------------------------------------------------------ sql (local)

_SQL_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum)\b", re.I
)


def sql_query(sql: str, params: tuple = (), limit: int = 200) -> list[dict]:
    """Read-only query against the local agent DB."""
    from backend.db import query as dbq

    if _SQL_FORBIDDEN.search(sql):
        raise PermissionError("read-only: write statements are not allowed")
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        raise PermissionError("only SELECT/WITH statements are allowed")
    if " limit " not in sql.lower():
        sql = f"{sql.rstrip().rstrip(';')} LIMIT {int(limit)}"
    return [dict(r) for r in dbq(sql, params)]


def hash_text(text: str, algo: str = "sha1") -> str:
    return hashlib.new(algo, (text or "").encode("utf-8", "ignore")).hexdigest()


def system_metrics() -> dict:
    import psutil

    vm = psutil.virtual_memory()
    p = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_percent": vm.percent,
        "ram_available_gb": round(vm.available / 1e9, 2),
        "process_rss_mb": round(p.memory_info().rss / 1e6, 1),
        "uptime_s": round(time.time() - p.create_time(), 1),
    }
