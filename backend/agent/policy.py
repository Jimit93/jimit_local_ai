"""Safety policy for everything the coding agent does to the machine.

Every workspace action passes through here first:

- paths are resolved (symlinks included) and must stay inside a registered workspace;
- secret files are never read, and secret-looking strings are redacted before any
  text reaches a model, a log or the UI;
- commands run as argv lists (never through a shell), only from an allowlist, with a
  scrubbed environment and no network-capable subcommands;
- destructive actions need an explicit approval that the agent loop cannot grant itself;
- every decision is written to `agent_audit`.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from backend.config import DATA_DIR, SETTINGS

RISK_SAFE, RISK_MEDIUM, RISK_HIGH, RISK_FORBIDDEN = "safe", "medium", "high", "forbidden"


class PolicyError(PermissionError):
    """Raised when an action is denied. The message is shown to the user as-is."""


class ApprovalRequired(PermissionError):
    """Raised when an action is allowed only with an explicit user approval."""

    def __init__(self, action: str, target: str, reason: str):
        super().__init__(f"approval required for {action} on {target}: {reason}")
        self.action, self.target, self.reason = action, target, reason


@dataclass
class Decision:
    allowed: bool
    risk: str
    reason: str
    needs_approval: bool = False


# ------------------------------------------------------------------ workspaces

_runtime_roots: list[str] = []

# A workspace may never be one of these (or a parent of the whole profile).
_FORBIDDEN_ROOTS = [
    os.environ.get("SystemRoot", r"C:\Windows"),
    os.environ.get("ProgramFiles", r"C:\Program Files"),
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    os.environ.get("ProgramData", r"C:\ProgramData"),
    os.environ.get("APPDATA", ""),
    os.environ.get("LOCALAPPDATA", ""),
]
_FORBIDDEN_NAMES = {".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker"}

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next", ".idea", ".vscode",
    ".tox", "site-packages", ".cache", "coverage", ".jla_trash",
}


def _norm(p: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def _is_within(child: str | Path, parent: str | Path) -> bool:
    c, p = _norm(child), _norm(parent)
    try:
        return os.path.commonpath([c, p]) == p
    except ValueError:  # different drives
        return False


def workspace_roots() -> list[str]:
    seen, out = set(), []
    for r in list(SETTINGS.agent.workspace_roots) + _runtime_roots:
        n = _norm(r)
        if n not in seen:
            seen.add(n)
            out.append(str(Path(r).resolve()))
    return out


def validate_workspace_root(path: str | Path) -> Path:
    """A directory may become a workspace only if it is a real, non-system project dir."""
    p = Path(path).expanduser()
    if not p.is_dir():
        raise PolicyError(f"not a directory: {path}")
    p = p.resolve()
    if p.parent == p:
        raise PolicyError("a drive root cannot be a workspace")
    home = Path.home().resolve()
    if _norm(p) == _norm(home) or _is_within(home, p):
        raise PolicyError("the whole user profile cannot be a workspace; pick a project folder")
    import tempfile

    in_temp = _is_within(p, tempfile.gettempdir()) and _norm(p) != _norm(tempfile.gettempdir())
    for f in _FORBIDDEN_ROOTS:
        # %TEMP% lives under AppData but is scratch space, not configuration or credentials.
        if f and _is_within(p, f) and not in_temp:
            raise PolicyError(f"system location cannot be a workspace: {p}")
    if _FORBIDDEN_NAMES & {part.lower() for part in p.parts}:
        raise PolicyError("credential directories cannot be a workspace")
    return p


def register_workspace(path: str | Path) -> str:
    p = validate_workspace_root(path)
    if all(_norm(p) != _norm(r) for r in workspace_roots()):
        _runtime_roots.append(str(p))
    return str(p)


def resolve_workspace(workspace: str | Path | None) -> Path:
    """Return the workspace root. It must equal or sit inside a registered root."""
    ws = Path(workspace).resolve() if workspace else Path(workspace_roots()[0])
    if not ws.is_dir():
        raise PolicyError(f"workspace does not exist: {ws}")
    if not any(_is_within(ws, r) for r in workspace_roots()):
        raise PolicyError(f"workspace is not registered: {ws}")
    return ws


def resolve_path(workspace: str | Path | None, rel: str | Path, *, for_write: bool = False) -> Path:
    """Resolve `rel` against the workspace and refuse anything that escapes it."""
    ws = resolve_workspace(workspace)
    raw = Path(rel)
    target = (raw if raw.is_absolute() else ws / raw).resolve()
    if not _is_within(target, ws):
        raise PolicyError(f"path escapes the workspace: {rel}")
    parts = {p.lower() for p in target.relative_to(ws).parts}
    if _FORBIDDEN_NAMES & parts:
        raise PolicyError(f"credential directory is off-limits: {rel}")
    if for_write:
        if ".git" in parts:
            raise PolicyError("the agent never writes inside .git")
        if _is_within(target, DATA_DIR):
            raise PolicyError("the agent never writes into its own data directory")
    return target


def rel_path(workspace: str | Path, p: str | Path) -> str:
    return Path(p).resolve().relative_to(resolve_workspace(workspace)).as_posix()


# ---------------------------------------------------------------------- secrets

SECRET_FILE_GLOBS = [
    ".env", ".env.*", "*.pem", "*.key", "*.pfx", "*.p12", "*.keystore", "*.jks",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", ".npmrc", ".pypirc", ".netrc",
    "*.kdbx", "credentials", "credentials.*", "secrets.*", "*.secret", "service-account*.json",
]
_ALLOWED_ENV_TEMPLATES = {".env.example", ".env.sample", ".env.template"}

_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    # Names containing password/secret/credential: any quoted value of 8+ chars.
    re.compile(r"(?i)\b([\w\-]*(?:passw(?:or)?d|secret|credential)[\w\-]*)\s*[:=]\s*"
               r"(['\"])[^'\"\s]{8,}\2"),
    # Names containing key/token/auth: only credential-shaped values (16+ chars with
    # letters and digits), so `tokenizer = "bert-base-uncased"` is left intact.
    re.compile(r"(?i)\b([\w\-]*(?:api[_-]?key|access[_-]?key|token|auth)[\w\-]*)\s*[:=]\s*"
               r"(['\"])(?=[^'\"]*\d)(?=[^'\"]*[A-Za-z])[^'\"\s]{16,}\2"),
]


def redact_obj(obj, _depth: int = 0):
    """Redact every string inside a JSON-like structure. Must run before serialisation:
    once dumped, quotes are escaped and the value patterns no longer match."""
    if _depth > 20:
        return obj
    if isinstance(obj, str):
        return redact(obj)[0]
    if isinstance(obj, dict):
        return {k: redact_obj(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v, _depth + 1) for v in obj]
    return obj


def is_secret_file(path: str | Path) -> bool:
    name = Path(path).name.lower()
    if name in _ALLOWED_ENV_TEMPLATES:
        return False
    return any(fnmatch.fnmatch(name, g) for g in SECRET_FILE_GLOBS)


def redact(text: str) -> tuple[str, int]:
    """Replace secret-looking substrings. Returns (text, number of redactions)."""
    if not text:
        return text, 0
    n = 0
    out = text
    for rx in _SECRET_PATTERNS:
        def _sub(m: re.Match) -> str:
            nonlocal n
            n += 1
            if m.lastindex and m.lastindex >= 2 and m.group(2) in ("'", '"'):
                return f"{m.group(1)}={m.group(2)}[REDACTED]{m.group(2)}"
            return "[REDACTED]"
        out = rx.sub(_sub, out)
    return out, n


# --------------------------------------------------------------------- commands

# Allowlist, not denylist: anything not named below (curl, powershell, rm, pip, ...) is refused.
_GIT_READ_ONLY = {"status", "diff", "log", "show", "rev-parse", "ls-files", "blame",
                  "branch", "describe", "shortlog", "grep"}
_PY_MODULES = {"pytest": RISK_MEDIUM, "pyflakes": RISK_SAFE, "py_compile": RISK_SAFE,
               "unittest": RISK_MEDIUM, "compileall": RISK_SAFE, "black": RISK_MEDIUM,
               "ruff": RISK_MEDIUM}
_NPM_SCRIPTS = {"build", "test", "lint", "typecheck", "type-check", "check"}


def _exe_name(arg0: str) -> str:
    n = Path(arg0).name.lower()
    for suf in (".exe", ".cmd", ".bat"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def check_command(argv: list[str], workspace: str | Path | None = None) -> Decision:
    """Allowlist check. Argv only - shells, pipes and inline code are refused."""
    if not argv or not all(isinstance(a, str) for a in argv):
        return Decision(False, RISK_FORBIDDEN, "command must be a non-empty list of strings")
    exe, args = _exe_name(argv[0]), argv[1:]
    if any(ch in a for a in argv for ch in ("|", "&&", ";", "`", "$(")) and exe != "git":
        return Decision(False, RISK_FORBIDDEN, "shell operators are not allowed")

    if exe in ("python", "python3", "py"):
        if not args:
            return Decision(False, RISK_FORBIDDEN, "interactive interpreter not allowed")
        if args[0] in ("-c", "-i") or "-c" in args[:2]:
            return Decision(False, RISK_FORBIDDEN, "inline python code is not allowed")
        if args[0] == "-m":
            mod = args[1] if len(args) > 1 else ""
            if mod in _PY_MODULES:
                return Decision(True, _PY_MODULES[mod], f"python -m {mod}")
            return Decision(False, RISK_FORBIDDEN, f"python module not allowed: {mod}")
        script = args[0]
        if workspace is not None:
            try:
                p = resolve_path(workspace, script)
            except PolicyError as e:
                return Decision(False, RISK_FORBIDDEN, str(e))
            if p.suffix == ".py" and p.is_file():
                return Decision(True, RISK_HIGH, "runs a project script", needs_approval=True)
        return Decision(False, RISK_FORBIDDEN, "python may only run allowed modules or a workspace script")

    if exe == "git":
        sub = next((a for a in args if not a.startswith("-")), "")
        if sub not in _GIT_READ_ONLY:
            return Decision(False, RISK_FORBIDDEN, f"git {sub or '(none)'} is not read-only")
        if sub == "branch" and any(a in ("-d", "-D", "-m", "-M", "--delete", "--move") for a in args):
            return Decision(False, RISK_FORBIDDEN, "branch modification is not allowed")
        if "--output" in " ".join(args):
            return Decision(False, RISK_FORBIDDEN, "git output redirection is not allowed")
        return Decision(True, RISK_SAFE, f"git {sub} (read-only)")

    if exe == "npm":
        if args[:1] == ["test"]:
            return Decision(True, RISK_MEDIUM, "npm test")
        if args[:1] == ["run"] and len(args) > 1 and args[1] in _NPM_SCRIPTS:
            return Decision(True, RISK_MEDIUM, f"npm run {args[1]}")
        return Decision(False, RISK_FORBIDDEN, "npm may only run build/test/lint/typecheck scripts")

    if exe == "node":
        if not args or args[0] in ("-e", "--eval", "-p", "--print", "-i"):
            return Decision(False, RISK_FORBIDDEN, "inline node code is not allowed")
        tool = args[0].replace("\\", "/")
        if tool.endswith(("typescript/bin/tsc", "vite/bin/vite.js")):
            return Decision(True, RISK_MEDIUM, "node build tool")
        return Decision(False, RISK_FORBIDDEN, "node may only run tsc or vite from node_modules")

    return Decision(False, RISK_FORBIDDEN, f"executable not allowed: {exe}")


def resolve_executable(argv: list[str]) -> list[str]:
    """Map allowlisted names to concrete binaries without going through a shell."""
    exe = _exe_name(argv[0])
    if exe in ("python", "python3", "py"):
        return [sys.executable, *argv[1:]]
    if exe == "npm":
        node = shutil.which("node")
        if node:
            cli = Path(node).parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
            if cli.exists():
                return [node, str(cli), *argv[1:]]
        raise PolicyError("npm is not available without a shell on this machine")
    found = shutil.which(argv[0])
    if not found:
        raise PolicyError(f"executable not found: {argv[0]}")
    return [found, *argv[1:]]


_ENV_KEEP = re.compile(
    r"(?i)^(?:PATH|PATHEXT|SYSTEMROOT|SYSTEMDRIVE|WINDIR|COMSPEC|TEMP|TMP|USERPROFILE|HOMEDRIVE|"
    r"HOMEPATH|USERNAME|APPDATA|LOCALAPPDATA|PROGRAMDATA|PROGRAMFILES.*|COMMONPROGRAMFILES.*|"
    r"NUMBER_OF_PROCESSORS|PROCESSOR_.*|OS|LANG|LC_.*|PYTHON(?:PATH|IOENCODING|UTF8|HOME)|"
    r"VIRTUAL_ENV|CONDA_PREFIX|NODE_PATH|NODE_ENV|JLA_.*)$"
)
_ENV_SECRET = re.compile(r"(?i)(KEY|TOKEN|SECRET|PASSW|CREDENTIAL|AUTH|COOKIE|SESSION)")


def scrubbed_env(extra: dict | None = None) -> dict:
    """Environment for child processes: allowlisted names, secrets removed, offline."""
    env = {k: v for k, v in os.environ.items() if _ENV_KEEP.match(k) and not _ENV_SECRET.search(k)}
    env.update({
        "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PIP_NO_INDEX": "1",
        "npm_config_offline": "true", "NO_COLOR": "1",
    })
    env.update(extra or {})
    return env


# ------------------------------------------------------------ actions + approvals

ACTION_RISK = {
    "read": RISK_SAFE, "list": RISK_SAFE, "search": RISK_SAFE, "git_read": RISK_SAFE,
    "syntax_check": RISK_SAFE, "lint": RISK_SAFE, "process_monitor": RISK_SAFE,
    "write": RISK_MEDIUM, "edit": RISK_MEDIUM, "create": RISK_MEDIUM, "rename": RISK_MEDIUM,
    "run_tests": RISK_MEDIUM, "format": RISK_MEDIUM, "rollback": RISK_MEDIUM,
    "delete": RISK_HIGH, "run_script": RISK_HIGH, "kill_process": RISK_HIGH,
}
NEEDS_APPROVAL = {"delete", "run_script", "kill_process"}


def check_action(action: str, target: str, *, approved: bool = False) -> Decision:
    risk = ACTION_RISK.get(action)
    if risk is None:
        return Decision(False, RISK_FORBIDDEN, f"unknown action {action}")
    if action in NEEDS_APPROVAL and not approved:
        return Decision(False, risk, "needs explicit user approval", needs_approval=True)
    return Decision(True, risk, "allowed")


def enforce(action: str, target: str, *, approved: bool = False, task_id: str | None = None) -> Decision:
    d = check_action(action, target, approved=approved)
    audit(action, target, "allow" if d.allowed else ("approval_required" if d.needs_approval else "deny"),
          d.risk, d.reason, task_id=task_id)
    if d.needs_approval:
        raise ApprovalRequired(action, target, d.reason)
    if not d.allowed:
        raise PolicyError(d.reason)
    return d


def audit(action: str, target: str | None, decision: str, risk: str, detail: str = "",
          *, task_id: str | None = None) -> None:
    from backend.db import execute

    safe_target, _ = redact(str(target or "")[:500])
    safe_detail, _ = redact(str(detail or "")[:1000])
    try:
        execute("INSERT INTO agent_audit(ts,task_id,action,target,decision,risk,detail) "
                "VALUES(?,?,?,?,?,?,?)",
                (time.time(), task_id, action, safe_target, decision, risk, safe_detail))
    except Exception:
        pass  # auditing must never break the action it records


def recent_audit(limit: int = 100) -> list[dict]:
    from backend.db import query

    return [dict(r) for r in query("SELECT * FROM agent_audit ORDER BY id DESC LIMIT ?", (limit,))]
