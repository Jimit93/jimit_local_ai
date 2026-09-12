import math

import pytest

from backend.tools import deterministic as D
from backend.tools import registry as R


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("25 * 40", 1000),
        ("What is 25 * 40?", 1000),
        ("(120 + 30) / 5", 30),
        ("2^10", 1024),
        ("17 % 5", 2),
        ("sqrt(144)", 12),
        ("15% of 200", 30.0),
        ("1,250 * 4", 5000),
        ("7 times 8", 56),
        ("100 divided by 4", 25),
        ("round(3.14159, 2)", 3.14),
    ],
)
def test_calculate(expr, expected):
    assert D.calculate(expr)["value"] == pytest.approx(expected)


def test_calculate_rejects_code():
    with pytest.raises(Exception):
        D.calculate("__import__('os').system('dir')")


def test_looks_like_math_gate():
    assert D.looks_like_math("what is 25*40")
    assert D.looks_like_math("15% of 200")
    assert not D.looks_like_math("write a linkedin post about 5 lessons")
    assert not D.looks_like_math("summarise my last 3 posts")


def test_text_stats():
    t = "Hello world! This is a test.\n\nSecond paragraph here #ai #local https://x.com 🚀"
    s = D.text_stats(t)
    assert s["word_count"] == 14
    assert s["paragraph_count"] == 2
    assert s["hashtag_count"] == 2 and "#ai" in s["hashtags"]
    assert s["url_count"] == 1
    assert s["emoji_count"] == 1
    assert s["exclamation_count"] == 1
    assert 0 < s["type_token_ratio"] <= 1


def test_sentences_and_paragraphs():
    assert len(D.sentences("A one. B two! C three?")) == 3
    assert len(D.paragraphs("a\n\nb\n\nc")) == 3


def test_dedupe():
    out = D.dedupe(["A b", "a  B ", "c"])
    assert out["unique_count"] == 2 and out["duplicate_count"] == 1


def test_filter_sort_aggregate():
    recs = [{"n": "3", "g": "x"}, {"n": "10", "g": "y"}, {"n": "7", "g": "x"}]
    assert len(D.filter_records(recs, "n", "gt", 5)) == 2
    assert D.sort_records(recs, "n", reverse=True)[0]["n"] == "10"
    assert D.aggregate(recs, "n", "sum") == 20
    assert D.aggregate(recs, "n", "sum", group_by="g") == {"x": 10.0, "y": 10.0}
    assert D.aggregate(recs, "n", "count") == 3


def test_date_tools():
    assert D.date_diff("2026-01-01", "2026-01-31")["days"] == 30
    assert D.date_add("2026-01-01", days=1)["iso"].startswith("2026-01-02")
    assert D.now_info()["weekday"]


def test_describe():
    d = D.describe([1, 2, 3, 4])
    assert d["n"] == 4 and d["mean"] == 2.5 and d["median"] == 2.5


def test_sql_readonly_guard():
    with pytest.raises(PermissionError):
        D.sql_query("DELETE FROM linkedin_documents")
    with pytest.raises(PermissionError):
        D.sql_query("DROP TABLE topics")


def test_registry_call_ok_and_error():
    ok = R.call("calculate", expression="6*7")
    assert ok.ok and ok.result["value"] == 42 and ok.confidence == 1.0
    assert ok.llm_used is False
    bad = R.call("calculate", expression="not math")
    assert not bad.ok and bad.confidence == 0.0
    missing = R.call("nope")
    assert not missing.ok


def test_registry_has_core_tools():
    names = {t.name for t in R.all_tools()}
    for n in ("calculate", "text_stats", "sql_query", "regex_search", "aggregate"):
        assert n in names
