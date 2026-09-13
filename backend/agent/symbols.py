"""Symbol and import extraction. Deterministic; no model.

Python is parsed with `ast` (exact line ranges). TypeScript/JavaScript use anchored
regexes plus brace matching for the end line - approximate, and marked as such.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass

LANG_BY_SUFFIX = {
    ".py": "python", ".ts": "typescript", ".tsx": "typescript", ".js": "javascript",
    ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".json": "json",
    ".md": "markdown", ".css": "css", ".html": "html", ".toml": "toml", ".yaml": "yaml",
    ".yml": "yaml", ".ini": "ini", ".cfg": "ini", ".txt": "text", ".bat": "batch",
    ".ps1": "powershell", ".sh": "shell", ".sql": "sql", ".rs": "rust", ".go": "go",
    ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp",
}


def language_of(path: str) -> str | None:
    p = path.lower()
    for suf, lang in LANG_BY_SUFFIX.items():
        if p.endswith(suf):
            return lang
    return None


@dataclass
class Symbol:
    name: str
    kind: str          # function|method|class|constant|interface|type|enum|component
    line: int
    end_line: int
    parent: str | None = None
    signature: str = ""
    doc: str = ""
    approx: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ python

def _py_signature(node: ast.AST) -> str:
    try:
        args = ast.unparse(node.args)  # type: ignore[attr-defined]
    except Exception:
        args = "..."
    ret = ""
    if getattr(node, "returns", None) is not None:
        try:
            ret = " -> " + ast.unparse(node.returns)  # type: ignore[attr-defined]
        except Exception:
            ret = ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({args}){ret}"[:240]  # type: ignore[attr-defined]


def _first_doc_line(node: ast.AST) -> str:
    try:
        d = ast.get_docstring(node)  # type: ignore[arg-type]
    except Exception:
        d = None
    return (d or "").strip().splitlines()[0][:200] if d else ""


def python_symbols(source: str) -> tuple[list[Symbol], list[tuple[str, int]], str | None]:
    """Return (symbols, imports, syntax_error)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [], [], f"SyntaxError: {e.msg} (line {e.lineno})"
    syms: list[Symbol] = []
    imports: list[tuple[str, int]] = []

    def visit(body: list[ast.stmt], parent: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                syms.append(Symbol(node.name, "method" if parent else "function", node.lineno,
                                   getattr(node, "end_lineno", node.lineno) or node.lineno,
                                   parent, _py_signature(node), _first_doc_line(node)))
            elif isinstance(node, ast.ClassDef):
                syms.append(Symbol(node.name, "class", node.lineno,
                                   getattr(node, "end_lineno", node.lineno) or node.lineno,
                                   parent, f"class {node.name}", _first_doc_line(node)))
                visit(node.body, node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and parent is None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and t.id.isupper() and len(t.id) > 1:
                        syms.append(Symbol(t.id, "constant", node.lineno,
                                           getattr(node, "end_lineno", node.lineno) or node.lineno))
            elif isinstance(node, ast.Import):
                imports.extend((a.name, node.lineno) for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mod = ("." * node.level) + (node.module or "")
                imports.append((mod, node.lineno))
            elif isinstance(node, (ast.If, ast.Try)) and parent is None:
                # Conditional imports / definitions at module level.
                visit(getattr(node, "body", []), None)

    visit(tree.body, None)
    for node in ast.walk(tree):  # function-local imports (used by this repo on purpose)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.ImportFrom):
                    imports.append((("." * sub.level) + (sub.module or ""), sub.lineno))
                elif isinstance(sub, ast.Import):
                    imports += [(a.name, sub.lineno) for a in sub.names]
    return syms, sorted(set(imports), key=lambda t: t[1]), None


# --------------------------------------------------------------- ts / js

_TS_PATTERNS = [
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)\s*(\([^)]*\))?"), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
    (re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"), "interface"),
    (re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*(?:<[^=]*>)?\s*="), "type"),
    (re.compile(r"^\s*(?:export\s+)?(?:const\s+)?enum\s+([A-Za-z_$][\w$]*)"), "enum"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*"
                r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=>"), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?function\b"), "function"),
]
_TS_IMPORT = re.compile(r"""^\s*import\s+(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)""")


def _block_end(lines: list[str], start: int) -> int:
    """Brace matching from a declaration line; strings/comments are stripped crudely."""
    depth, opened = 0, False
    for i in range(start, min(len(lines), start + 2000)):
        s = re.sub(r"(['\"`])(?:\\.|(?!\1).)*\1|//.*$", "", lines[i])
        for ch in s:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return i + 1
        if not opened and i > start and s.strip().endswith(";"):
            return i + 1
    return start + 1


def ts_symbols(source: str) -> tuple[list[Symbol], list[tuple[str, int]]]:
    lines = source.splitlines()
    syms: list[Symbol] = []
    imports: list[tuple[str, int]] = []
    for i, line in enumerate(lines):
        m = _TS_IMPORT.search(line)
        if m:
            imports.append((m.group(1) or m.group(2), i + 1))
            continue
        for rx, kind in _TS_PATTERNS:
            m = rx.match(line)
            if m:
                name = m.group(1)
                k = "component" if kind == "function" and name[:1].isupper() else kind
                syms.append(Symbol(name, k, i + 1, _block_end(lines, i), None,
                                   line.strip()[:240], "", approx=True))
                break
    return syms, imports


def extract(path: str, source: str) -> dict:
    lang = language_of(path)
    if lang == "python":
        syms, imports, err = python_symbols(source)
    elif lang in ("typescript", "javascript"):
        syms, imports = ts_symbols(source)
        err = None
    else:
        syms, imports, err = [], [], None
    return {"lang": lang, "symbols": syms, "imports": imports, "syntax_error": err}
