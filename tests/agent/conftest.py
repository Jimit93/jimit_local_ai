"""Agent tests run against a throwaway database and throwaway workspaces.

The real `data/agent.db` holds the user's corpus and history; nothing in these tests
may write to it.
"""
import textwrap
from pathlib import Path

import pytest

from backend import db as DB


@pytest.fixture(autouse=True)
def isolated_db(tmp_path_factory, monkeypatch):
    path = tmp_path_factory.mktemp("agentdb") / "agent_test.db"
    old = getattr(DB._local, "conn", None)
    DB._local.conn = None
    monkeypatch.setattr(DB, "DB_PATH", path)
    from backend.agent import checkpoints as CP

    monkeypatch.setattr(CP, "TRASH_DIR", path.parent / "trash")
    yield path
    conn = getattr(DB._local, "conn", None)
    if conn is not None:
        conn.close()
    DB._local.conn = old


SAMPLE_FILES = {
    "calc.py": '''\
        """Tiny calculator module."""


        def add(a, b):
            """Return the sum."""
            return a - b


        def mul(a, b):
            return a * b


        class Accumulator:
            def __init__(self):
                self.total = 0

            def push(self, x):
                self.total = add(self.total, x)
                return self.total
        ''',
    "tests/test_calc.py": '''\
        from calc import add, mul


        def test_add():
            assert add(2, 3) == 5


        def test_mul():
            assert mul(4, 5) == 20
        ''',
    "pytest.ini": "[pytest]\naddopts = -q\n",
    "config.json": '{"name": "sample", "debug": false}\n',
    ".env": "API_KEY=sk-live-abcdefghijklmnopqrstuvwxyz123456\n",
    "notes/readme.md": "# Sample\nThe accumulator uses add().\n",
    "node_modules/pkg/index.js": "module.exports = 1;\n",
}


def make_project(root: Path, files: dict | None = None) -> Path:
    for rel, body in (files or SAMPLE_FILES).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")
    return root


@pytest.fixture
def workspace(tmp_path):
    from backend.agent import policy

    root = make_project(tmp_path / "proj")
    policy.register_workspace(root)
    return str(root.resolve())
