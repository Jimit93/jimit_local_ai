"""Incremental project index and file-relevance ranking.

Build: walk the workspace; a file is re-read only when size/mtime changed, and
re-parsed only when its sha1 changed. Symbols and imports go to SQLite.

Rank: BM25 over an identifier-aware token stream per file (path, symbols, docstrings,
imports, head of content), fused with deterministic signals - an explicit file mention,
an exact symbol-name match, a traceback location, git-modified status and 1-hop import
neighbours of the strongest hit. Weights were chosen, then validated by
`evaluate_ranking`, whose ground truth is generated from the workspace at run time.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.agent import policy
from backend.agent.code_tools import iter_files
from backend.agent.symbols import LANG_BY_SUFFIX, extract, language_of
from backend.db import connect, query
from backend.nlp.keywords import BM25Index

INDEXABLE = set(LANG_BY_SUFFIX.values()) - {"text"}
MAX_INDEX_BYTES = 1_000_000
HEAD_CHARS = 12000
STALE_AFTER_S = 20.0

_CODE_STOP = {
    "self", "cls", "def", "return", "import", "from", "the", "and", "for", "with", "none",
    "true", "false", "const", "let", "var", "function", "export", "default", "class", "str",
    "int", "dict", "list", "bool", "float", "this", "that", "not", "are", "was", "has", "into",
    "of", "to", "in", "is", "it", "a", "an", "or", "if", "else", "elif", "try", "except", "as",
    "py", "ts", "tsx", "js", "file", "code", "where", "what", "how", "which", "does", "do",
    "please", "can", "you", "me", "my", "show", "find", "fix", "add", "make", "use", "used",
}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "ied", "ies", "ed", "es", "er", "s")


def _stem(w: str) -> str:
    """Light suffix stripping so 'escalate' meets 'escalation', 'routing' meets 'router'."""
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)].rstrip("e")
    return w.rstrip("e") if len(w) > 4 else w


def code_tokens(text: str) -> list[str]:
    """Identifiers plus their snake/camel parts, lower-cased and lightly stemmed."""
    out: list[str] = []
    for ident in _IDENT.findall(text or ""):
        low = ident.lower()
        parts = [p.lower() for chunk in ident.split("_") for p in _CAMEL.findall(chunk)]
        if len(low) > 2 and low not in _CODE_STOP:
            out.append(low)
            if len(parts) <= 1:
                st = _stem(low)
                if st != low:
                    out.append(st)
        if len(parts) > 1:
            for p in parts:
                if len(p) > 2 and p not in _CODE_STOP:
                    out.append(p)
                    st = _stem(p)
                    if st != p:
                        out.append(st)
    return out


def _identifier_like(word: str) -> bool:
    """snake_case, camelCase, PascalCase or dotted - not a plain English word."""
    return "_" in word or bool(re.search(r"[a-z][A-Z]", word)) or (word[:1].isupper() and any(c.isupper() for c in word[1:]))


# ------------------------------------------------------------------- building

_build_lock = threading.Lock()
_last_build: dict[str, float] = {}
_rankers: dict[str, "_Ranker"] = {}


def _pause_if_overloaded(cancel_event=None) -> int:
    """Indexing is background work: yield while the governor reports high load."""
    try:
        from backend.agent.governor import CAUTION, NORMAL, governor

        g = governor()
        if not g.running:
            return 0
        waited = 0
        while g.state not in (NORMAL, CAUTION) and waited < 60:
            if cancel_event is not None and cancel_event.is_set():
                break
            time.sleep(0.5)
            waited += 1
        return waited
    except Exception:
        return 0


def build(workspace: str, *, max_files: int = 20000, force: bool = False, cancel_event=None) -> dict:
    ws = str(policy.resolve_workspace(workspace))
    t0 = time.perf_counter()
    with _build_lock:
        conn = connect()
        existing = {r["path"]: (r["size"], r["mtime"], r["sha1"]) for r in
                    query("SELECT path,size,mtime,sha1 FROM code_files WHERE workspace=?", (ws,))}
        seen: set[str] = set()
        changed = unchanged = skipped = paused_s = 0
        parse_errors: list[dict] = []
        for i, (rel, p) in enumerate(iter_files(ws, max_files=max_files)):
            if i and i % 250 == 0:
                paused_s += _pause_if_overloaded(cancel_event) * 0.5
                if cancel_event is not None and cancel_event.is_set():
                    break
            lang = language_of(rel)
            if lang not in INDEXABLE or policy.is_secret_file(p):
                skipped += 1
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > MAX_INDEX_BYTES:
                skipped += 1
                continue
            seen.add(rel)
            prev = existing.get(rel)
            if prev and not force and prev[0] == st.st_size and abs(prev[1] - st.st_mtime) < 1e-6:
                unchanged += 1
                continue
            try:
                data = p.read_bytes()
            except OSError:
                continue
            sha = hashlib.sha1(data).hexdigest()
            if prev and not force and prev[2] == sha:
                conn.execute("UPDATE code_files SET mtime=?, indexed_at=? WHERE workspace=? AND path=?",
                             (st.st_mtime, time.time(), ws, rel))
                unchanged += 1
                continue
            text = data.decode("utf-8", errors="replace")
            info = extract(rel, text)
            if info["syntax_error"]:
                parse_errors.append({"path": rel, "error": info["syntax_error"]})
            summary = _summary(text, lang)
            conn.execute("DELETE FROM code_symbols WHERE workspace=? AND path=?", (ws, rel))
            conn.execute("DELETE FROM code_imports WHERE workspace=? AND path=?", (ws, rel))
            conn.executemany(
                "INSERT INTO code_symbols(workspace,path,name,kind,line,end_line,parent,signature) "
                "VALUES(?,?,?,?,?,?,?,?)",
                [(ws, rel, s.name, s.kind, s.line, s.end_line, s.parent,
                  (s.signature + (" # " + s.doc if s.doc else ""))[:400]) for s in info["symbols"]])
            conn.executemany("INSERT INTO code_imports(workspace,path,module,line) VALUES(?,?,?,?)",
                             [(ws, rel, m, ln) for m, ln in info["imports"]])
            conn.execute(
                "INSERT INTO code_files(workspace,path,size,mtime,sha1,lang,lines,summary,indexed_at) "
                "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(workspace,path) DO UPDATE SET size=excluded.size, "
                "mtime=excluded.mtime, sha1=excluded.sha1, lang=excluded.lang, lines=excluded.lines, "
                "summary=excluded.summary, indexed_at=excluded.indexed_at",
                (ws, rel, st.st_size, st.st_mtime, sha, lang, text.count("\n") + 1, summary, time.time()))
            changed += 1
            if changed % 500 == 0:
                conn.commit()
        removed = [p for p in existing if p not in seen]
        for rel in removed:
            for t in ("code_files", "code_symbols", "code_imports"):
                conn.execute(f"DELETE FROM {t} WHERE workspace=? AND path=?", (ws, rel))
        conn.commit()
        _last_build[ws] = time.time()
        if changed or removed:
            _rankers.pop(ws, None)
    return {"workspace": ws, "files": len(seen), "changed": changed, "unchanged": unchanged,
            "removed": len(removed), "skipped": skipped, "parse_errors": parse_errors[:20],
            "paused_s": paused_s, "ms": round((time.perf_counter() - t0) * 1000, 1)}


def ensure_index(workspace: str, *, max_age_s: float = STALE_AFTER_S) -> dict | None:
    ws = str(policy.resolve_workspace(workspace))
    if time.time() - _last_build.get(ws, 0) < max_age_s:
        return None
    return build(ws)


def _summary(text: str, lang: str | None) -> str:
    if lang == "python":
        m = re.match(r'\s*(?:#.*\n\s*)*(?:"""|\'\'\')\s*(.+)', text)
        if m:
            return m.group(1).strip()[:200]
    m = re.match(r"\s*(?://|#|/\*+|\*)\s*(.+)", text)
    return m.group(1).strip()[:200] if m else ""


# -------------------------------------------------------------------- queries

def find_symbols(workspace: str, name: str, *, kind: str | None = None, exact: bool = False,
                 limit: int = 100) -> dict:
    ws = str(policy.resolve_workspace(workspace))
    ensure_index(ws)
    sql = "SELECT path,name,kind,line,end_line,parent,signature FROM code_symbols WHERE workspace=? AND "
    if exact:
        sql += "name = ? COLLATE NOCASE"
        params: list = [ws, name]
    else:
        sql += "name LIKE ? ESCAPE '\\'"
        params = [ws, "%" + name.replace("%", r"\%").replace("_", r"\_") + "%"]
    if kind:
        sql += " AND kind=?"
        params.append(kind)
    sql += " ORDER BY (lower(name)=lower(?)) DESC, length(name), path LIMIT ?"
    params += [name, limit]
    rows = [dict(r) for r in query(sql, params)]
    return {"name": name, "count": len(rows), "symbols": rows, "source": "index"}


def file_record(workspace: str, rel: str) -> dict | None:
    ws = str(policy.resolve_workspace(workspace))
    rows = query("SELECT * FROM code_files WHERE workspace=? AND path=?", (ws, rel))
    return dict(rows[0]) if rows else None


def symbols_in(workspace: str, rel: str) -> list[dict]:
    ws = str(policy.resolve_workspace(workspace))
    return [dict(r) for r in query(
        "SELECT name,kind,line,end_line,parent,signature FROM code_symbols WHERE workspace=? AND path=? "
        "ORDER BY line", (ws, rel))]


def import_graph(workspace: str) -> dict[str, set[str]]:
    """Undirected file-level graph from resolvable local imports (python dotted, ts relative)."""
    ws = str(policy.resolve_workspace(workspace))
    files = {r["path"] for r in query("SELECT path FROM code_files WHERE workspace=?", (ws,))}
    by_module: dict[str, str] = {}
    for f in files:
        if f.endswith(".py"):
            mod = f[:-3].replace("/", ".")
            by_module[mod] = f
            if mod.endswith(".__init__"):
                by_module[mod[: -len(".__init__")]] = f
    graph: dict[str, set[str]] = {f: set() for f in files}
    for r in query("SELECT path,module FROM code_imports WHERE workspace=?", (ws,)):
        src, mod = r["path"], r["module"]
        target = None
        if src.endswith(".py"):
            if mod.startswith("."):
                level = len(mod) - len(mod.lstrip("."))
                base = src.split("/")[:-level] if level else src.split("/")[:-1]
                mod = ".".join(base + [m for m in [mod.lstrip(".")] if m])
            parts = mod.split(".")
            for k in range(len(parts), 0, -1):
                cand = ".".join(parts[:k])
                if cand in by_module:
                    target = by_module[cand]
                    break
        elif mod.startswith("."):
            base = Path(src).parent
            stem = (base / mod).as_posix()
            stem = re.sub(r"(^|/)\./", r"\1", stem)
            parts: list[str] = []
            for seg in stem.split("/"):
                if seg == "..":
                    if parts:
                        parts.pop()
                elif seg and seg != ".":
                    parts.append(seg)
            stem = "/".join(parts)
            for ext in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"):
                if stem + ext in files:
                    target = stem + ext
                    break
        if target and target != src:
            graph.setdefault(src, set()).add(target)
            graph.setdefault(target, set()).add(src)
    return graph


# -------------------------------------------------------------------- ranking

@dataclass
class _Ranker:
    paths: list[str]
    bm25: BM25Index
    symbols_by_file: dict[str, set[str]]
    built_at: float = field(default_factory=time.time)


def _ranker(ws: str) -> _Ranker:
    r = _rankers.get(ws)
    if r is not None:
        return r
    files = [dict(x) for x in query("SELECT path,summary FROM code_files WHERE workspace=? ORDER BY path", (ws,))]
    syms: dict[str, list[dict]] = {}
    for s in query("SELECT path,name,signature FROM code_symbols WHERE workspace=?", (ws,)):
        syms.setdefault(s["path"], []).append(dict(s))
    imps: dict[str, list[str]] = {}
    for m in query("SELECT path,module FROM code_imports WHERE workspace=?", (ws,)):
        imps.setdefault(m["path"], []).append(m["module"])
    docs = []
    for f in files:
        p = Path(ws) / f["path"]
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:HEAD_CHARS]
        except OSError:
            head = ""
        path_part = " ".join([f["path"].replace("/", " ").replace(".", " ")] * 3)
        sym_part = " ".join(" ".join([s["name"], s["name"], s["signature"] or ""]) for s in syms.get(f["path"], []))
        docs.append(" ".join([path_part, sym_part, f["summary"] or "", " ".join(imps.get(f["path"], [])), head]))
    r = _Ranker([f["path"] for f in files], BM25Index(docs, tokenizer=code_tokens),
                {p: {s["name"].lower() for s in v} for p, v in syms.items()})
    _rankers[ws] = r
    return r


_FILE_MENTION = re.compile(r"[\w./\\-]+\.(?:py|tsx?|jsx?|json|md|css|html|toml|ya?ml|ini|cfg|sql|bat|ps1)\b", re.I)
_TB_FILE = re.compile(r'File "([^"]+)", line (\d+)|^([\w./\\-]+\.py):(\d+):', re.M)


def _strip_abs_paths(text: str, ws: str) -> str:
    if not text:
        return ""
    out = text
    for variant in {ws, ws.replace("\\", "/"), ws.replace("/", "\\")}:
        out = re.sub(re.escape(variant) + r"[\\/]?", "", out, flags=re.I)
    # Remaining absolute paths (site-packages, interpreter) carry no project signal.
    return re.sub(r"(?:[A-Za-z]:)?[\\/](?:[^\s\"':]+[\\/])+", "", out)


def mentioned_files(text: str) -> list[str]:
    return [m.group(0).replace("\\", "/").lstrip("./") for m in _FILE_MENTION.finditer(text or "")]


def traceback_locations(text: str, workspace: str | None = None) -> list[dict]:
    out = []
    ws = str(policy.resolve_workspace(workspace)).replace("\\", "/").lower() if workspace else None
    for m in _TB_FILE.finditer(text or ""):
        path = (m.group(1) or m.group(3) or "").replace("\\", "/")
        line = int(m.group(2) or m.group(4) or 0)
        if ws and path.lower().startswith(ws):
            path = path[len(ws):].lstrip("/")
        if "site-packages" in path or path.startswith("<"):
            continue
        loc = {"path": path, "line": line}
        if loc not in out:
            out.append(loc)
    return out


# name_match 0.8 with an IDF floor of 0.3 was chosen on identical generated case sets
# (this repo, 2026-09-13): docstring queries P@1 0.890 / MRR 0.936 (unchanged vs no
# name signal), module-reference queries P@1 0.183 -> 0.366, MRR 0.302 -> 0.468.
SIGNAL_WEIGHTS = {"file_mentioned": 1.0, "name_match": 0.8, "name_idf_floor": 0.3, "traceback": 0.8,
                  "traceback_innermost": 1.1, "git_modified": 0.1, "imports_top_hit": 0.15}


def rank_files(workspace: str, request: str, *, top_k: int = 8, error_text: str = "",
               modified: list[str] | None = None, use_graph: bool = True, weights: dict | None = None) -> dict:
    W = {**SIGNAL_WEIGHTS, **(weights or {})}
    ws = str(policy.resolve_workspace(workspace))
    ensure_index(ws)
    t0 = time.perf_counter()
    rk = _ranker(ws)
    if not rk.paths:
        return {"files": [], "confidence": 0.0, "ms": 0.0, "method": "empty index"}
    # Tracebacks repeat the absolute workspace path in every frame; its tokens
    # ("users", "temp", "pytest-of-...") would otherwise dominate BM25.
    clean_err = _strip_abs_paths(error_text[:4000], ws)
    raw = rk.bm25.normalised_search(request + " " + clean_err[:2000], top_k=max(top_k * 4, 30))
    scores: dict[str, dict] = {}
    for i, s in raw:
        scores[rk.paths[i]] = {"path": rk.paths[i], "bm25": s, "signals": []}

    def bump(path: str, amount: float, why: str) -> None:
        if path not in rk.paths:
            return
        e = scores.setdefault(path, {"path": path, "bm25": 0.0, "signals": []})
        e["signals"].append(why)
        e[why] = e.get(why, 0.0) + amount

    low_paths = {p.lower(): p for p in rk.paths}
    for m in mentioned_files(request + "\n" + error_text):
        ml = m.lower()
        for lp, orig in low_paths.items():
            if lp == ml or lp.endswith("/" + ml):
                bump(orig, W["file_mentioned"], "file_mentioned")
    # Only identifier-shaped words (or words written as calls / in backticks) count as
    # symbol references; "model" or "build" in a sentence is not a reference to `model()`.
    marked = set(re.findall(r"`([A-Za-z_][\w.]*)`|\b([A-Za-z_]\w*)\(", request))
    marked_names = {a or b for a, b in marked}
    # "the Governor class", "class Governor", "function run_task", "run_task method"
    for m in re.finditer(r"\b(?:class|function|method|def|component)\s+`?([A-Za-z_]\w*)|"
                         r"\b([A-Za-z_]\w*)`?\s+(?:class|function|method|component)\b", request):
        marked_names.add(m.group(1) or m.group(2))
    idents = {t.lower() for t in _IDENT.findall(request)
              if len(t) > 2 and t.lower() not in _CODE_STOP and (_identifier_like(t) or t in marked_names)}
    for path, names in rk.symbols_by_file.items():
        hits = idents & names
        if hits:
            bump(path, min(0.6, 0.35 + 0.1 * len(hits)), "symbol_match")
    # A word naming a module ("the governor", "router logic") points at that file.
    # Scaled by how distinctive the name is across the code (IDF): "governor" or "chunking"
    # identifies a module; "state" or "config" appears everywhere and identifies nothing.
    words = {_stem(w.lower()) for w in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", request) if w.lower() not in _CODE_STOP}
    max_idf = max(rk.bm25.idf.values(), default=1.0) or 1.0
    for path in rk.paths:
        stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
        if len(stem) >= 4 and _stem(stem) in words and W["name_match"]:
            idf = max(rk.bm25.idf.get(stem, 0.0), rk.bm25.idf.get(_stem(stem), 0.0))
            rarity = max(0.0, min(1.0, (idf / max_idf - W.get("name_idf_floor", 0.0)) / max(1e-6, 1 - W.get("name_idf_floor", 0.0))))
            if rarity > 0:
                bump(path, W["name_match"] * rarity, "name_match")
    locs = [loc for loc in traceback_locations(error_text, ws)
            if any(lp == loc["path"].lower() or lp.endswith("/" + loc["path"].lower()) for lp in low_paths)]
    for i, loc in enumerate(locs):
        for lp, orig in low_paths.items():
            if lp == loc["path"].lower() or lp.endswith("/" + loc["path"].lower()):
                # The innermost workspace frame is where the error was raised.
                bump(orig, W["traceback_innermost"] if i == len(locs) - 1 else W["traceback"], "traceback")
    for mp in modified or []:
        bump(mp, W["git_modified"], "git_modified")
    wants_tests = bool(re.search(r"\btests?\b|pytest|spec", request, re.I))
    wants_docs = bool(re.search(r"\b(?:docs?|readme|markdown|report|json|config|documentation|skill)\b", request, re.I))
    for e in scores.values():
        is_test = bool(re.search(r"(^|/)(tests?|__tests__)/|(^|/)test_|_test\.py$|\.test\.|\.spec\.", e["path"]))
        e["is_test"] = is_test
        e["score"] = e["bm25"] + sum(e.get(k, 0.0) for k in ("file_mentioned", "symbol_match", "name_match",
                                                             "traceback", "git_modified"))
        if is_test and not wants_tests:
            e["score"] *= 0.85
        lang = language_of(e["path"])
        if not wants_docs and lang in ("markdown", "json", "yaml", "text", "ini", "toml", "html", "css"):
            # Prose and data mention everything; for code questions source files come first.
            e["score"] *= 0.7
    ranked = sorted(scores.values(), key=lambda e: -e["score"])
    if use_graph and ranked and ranked[0]["score"] > 0:
        neighbours = import_graph(ws).get(ranked[0]["path"], set())
        for e in ranked[1:]:
            if e["path"] in neighbours:
                e["score"] += W["imports_top_hit"]
                e["signals"].append("imports_top_hit")
        ranked.sort(key=lambda e: -e["score"])
    top = ranked[:top_k]
    conf = 0.0
    if top:
        s1 = top[0]["score"]
        s2 = top[1]["score"] if len(top) > 1 else 0.0
        conf = round(min(1.0, 0.45 * min(1.0, s1) + 0.55 * ((s1 - s2) / s1 if s1 else 0.0)), 3)
        if top[0]["signals"]:
            conf = max(conf, 0.8)
    for e in top:
        e["score"] = round(e["score"], 4)
    return {"files": top, "confidence": conf, "ms": round((time.perf_counter() - t0) * 1000, 2),
            "method": "bm25(code tokens) + mention/symbol/traceback/import signals",
            "candidates": len(scores), "indexed_files": len(rk.paths)}


def repo_map(workspace: str, files: list[str] | None = None, *, max_chars: int = 2500) -> str:
    """Compact outline: file -> top-level symbols. Orients a model in few tokens."""
    ws = str(policy.resolve_workspace(workspace))
    ensure_index(ws)
    rows = query("SELECT path,summary FROM code_files WHERE workspace=? ORDER BY path", (ws,))
    wanted = set(files or [])
    lines: list[str] = []
    for r in rows:
        if wanted and r["path"] not in wanted:
            continue
        syms = [s for s in symbols_in(ws, r["path"]) if s["kind"] in ("class", "function", "component")]
        names = ", ".join(s["name"] for s in syms[:12]) + (" …" if len(syms) > 12 else "")
        lines.append(f"{r['path']}: {names}" + (f"  # {r['summary'][:80]}" if r["summary"] else ""))
        if sum(len(x) + 1 for x in lines) > max_chars:
            lines.append("…")
            break
    return "\n".join(lines)


# ------------------------------------------------------------------ evaluation

def evaluate_ranking(workspace: str, *, sample: int = 150, top_k: int = 5, seed: int = 7) -> dict:
    """Ground truth generated from the workspace itself, at run time.

    For each documented function/class, the query is its docstring with the symbol's
    own name removed (otherwise the exact-name signal makes the task trivial); the
    expected answer is the file that defines it. Reports P@1, Recall@k and MRR for
    BM25 alone and for the fused ranker.
    """
    import random

    ws = str(policy.resolve_workspace(workspace))
    build(ws)
    rows = [dict(r) for r in query(
        "SELECT path,name,signature FROM code_symbols WHERE workspace=? AND kind IN ('function','class','method') "
        "AND signature LIKE '% # %'", (ws,))]
    cases = []
    for r in rows:
        doc = r["signature"].split(" # ", 1)[1]
        q = re.sub(re.escape(r["name"]), " ", doc, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        if len(code_tokens(q)) >= 3:
            cases.append({"query": q, "expected": r["path"], "symbol": r["name"]})
    random.Random(seed).shuffle(cases)
    cases = cases[:sample]
    rk = _ranker(ws)

    def metrics(ranks: list[int | None]) -> dict:
        n = len(ranks) or 1
        return {"p_at_1": round(sum(1 for r in ranks if r == 1) / n, 3),
                f"recall_at_{top_k}": round(sum(1 for r in ranks if r and r <= top_k) / n, 3),
                "mrr": round(sum(1 / r for r in ranks if r) / n, 3)}

    bm_ranks, fused_ranks = [], []
    t0 = time.perf_counter()
    for c in cases:
        hits = [rk.paths[i] for i, _ in rk.bm25.search(c["query"], top_k=50)]
        bm_ranks.append(hits.index(c["expected"]) + 1 if c["expected"] in hits else None)
        fused = [e["path"] for e in rank_files(ws, c["query"], top_k=50)["files"]]
        fused_ranks.append(fused.index(c["expected"]) + 1 if c["expected"] in fused else None)
    ms = round((time.perf_counter() - t0) * 1000 / max(1, len(cases)) / 2, 2)

    # Second set: requests that name a module ("explain how the governor works"), for
    # source files whose name is unique in the workspace.
    from collections import Counter

    src = [p for p in rk.paths if language_of(p) in ("python", "typescript", "javascript")
           and not re.search(r"(^|/)tests?/|(^|/)test_", p)]
    stem = lambda p: p.rsplit("/", 1)[-1].rsplit(".", 1)[0]  # noqa: E731
    counts = Counter(stem(p).lower() for p in src)
    mod_cases = [{"query": f"explain how the {stem(p).replace('_', ' ')} works", "expected": p}
                 for p in src if counts[stem(p).lower()] == 1 and len(stem(p)) >= 4]
    mb, mf = [], []
    for c in mod_cases:
        hits = [rk.paths[i] for i, _ in rk.bm25.search(c["query"], top_k=50)]
        mb.append(hits.index(c["expected"]) + 1 if c["expected"] in hits else None)
        fused = [e["path"] for e in rank_files(ws, c["query"], top_k=50)["files"]]
        mf.append(fused.index(c["expected"]) + 1 if c["expected"] in fused else None)
    return {"workspace": ws, "cases": len(cases), "indexed_files": len(rk.paths),
            "bm25": metrics(bm_ranks), "fused": metrics(fused_ranks), "ms_per_query": ms,
            "module_cases": len(mod_cases), "module_bm25": metrics(mb), "module_fused": metrics(mf)}
