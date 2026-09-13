"""Safety policy: sandbox, secrets, command allowlist, approvals, audit."""
from pathlib import Path

import pytest

from backend.agent import policy as P


def test_paths_cannot_escape_the_workspace(workspace):
    assert P.resolve_path(workspace, "calc.py").name == "calc.py"
    for bad in ("../outside.txt", "..\\..\\Windows\\win.ini", "C:\\Windows\\win.ini"):
        with pytest.raises(P.PolicyError):
            P.resolve_path(workspace, bad)


def test_writes_into_git_and_data_dir_are_refused(workspace):
    (Path(workspace) / ".git").mkdir()
    with pytest.raises(P.PolicyError):
        P.resolve_path(workspace, ".git/config", for_write=True)


def test_unregistered_and_system_workspaces_are_refused(tmp_path):
    with pytest.raises(P.PolicyError):
        P.resolve_workspace(tmp_path / "never_registered_dir_xyz")
    with pytest.raises(P.PolicyError):
        P.validate_workspace_root("C:\\Windows")
    with pytest.raises(P.PolicyError):
        P.validate_workspace_root(Path.home())


def test_secret_files_and_redaction():
    assert P.is_secret_file(".env") and P.is_secret_file("id_rsa") and P.is_secret_file("key.pem")
    assert not P.is_secret_file(".env.example") and not P.is_secret_file("calc.py")
    text = ('db_password = "hunter2hunter2"\nkey=sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX\n'
            'GITHUB_AUTH_TOKEN = "a1b2c3d4e5f6g7h8i9j0"\ntokenizer = "bert-base-uncased"\nplain = 1')
    red, n = P.redact(text)
    assert n == 3
    for leaked in ("hunter2hunter2", "sk-proj-", "a1b2c3d4e5f6g7h8i9j0"):
        assert leaked not in red
    assert "bert-base-uncased" in red and "plain = 1" in red


@pytest.mark.parametrize("argv,allowed", [
    (["python", "-m", "pytest", "-q"], True),
    (["python", "-m", "pyflakes", "x.py"], True),
    (["git", "status"], True),
    (["git", "diff", "--stat"], True),
    (["npm", "run", "build"], True),
    (["python", "-c", "import os; os.remove('x')"], False),
    (["python", "-m", "pip", "install", "requests"], False),
    (["git", "push"], False),
    (["git", "reset", "--hard"], False),
    (["git", "branch", "-D", "main"], False),
    (["npm", "install", "left-pad"], False),
    (["curl", "http://example.com"], False),
    (["powershell", "-c", "Remove-Item x"], False),
    (["cmd", "/c", "del", "x"], False),
    (["node", "-e", "require('fs').rmSync('x')"], False),
    ([], False),
])
def test_command_allowlist(argv, allowed, workspace):
    assert P.check_command(argv, workspace).allowed is allowed


def test_project_script_needs_approval(workspace):
    d = P.check_command(["python", "calc.py"], workspace)
    assert d.needs_approval and d.risk == P.RISK_HIGH


def test_scrubbed_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    env = P.scrubbed_env()
    assert "OPENAI_API_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["HF_HUB_OFFLINE"] == "1" and "PATH" in {k.upper() for k in env}


def test_delete_needs_approval_and_everything_is_audited(workspace):
    with pytest.raises(P.ApprovalRequired):
        P.enforce("delete", "calc.py")
    P.enforce("delete", "calc.py", approved=True)
    decisions = [r["decision"] for r in P.recent_audit(5)]
    assert "approval_required" in decisions and "allow" in decisions
