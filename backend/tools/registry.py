"""Tool registry. The LLM (or the router) may choose a tool; the tool does the work."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from backend.common import ProcessorResult
from backend.tools import deterministic as D


@dataclass
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str
    args: str
    category: str


_TOOLS: dict[str, Tool] = {}


def register(name: str, fn: Callable, description: str, args: str, category: str) -> None:
    _TOOLS[name] = Tool(name, fn, description, args, category)


def get(name: str) -> Tool | None:
    return _TOOLS.get(name)


def all_tools() -> list[Tool]:
    return list(_TOOLS.values())


def describe_tools() -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "args": t.args, "category": t.category}
        for t in _TOOLS.values()
    ]


def call(name: str, **kwargs) -> ProcessorResult:
    """Run a registered tool and wrap the outcome in the standard envelope."""
    t = _TOOLS.get(name)
    t0 = time.perf_counter()
    if t is None:
        return ProcessorResult(
            result=None, confidence=0.0, processor="tool_registry",
            latency_ms=(time.perf_counter() - t0) * 1000, error=f"unknown tool: {name}"
        )
    try:
        out = t.fn(**kwargs)
        return ProcessorResult(
            result=out, confidence=1.0, processor=f"tool:{name}",
            latency_ms=(time.perf_counter() - t0) * 1000, layer="deterministic",
        )
    except Exception as e:
        return ProcessorResult(
            result=None, confidence=0.0, processor=f"tool:{name}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=f"{type(e).__name__}: {e}", layer="deterministic",
        )


def _bootstrap() -> None:
    r = register
    r("calculate", D.calculate, "Evaluate an arithmetic or percentage expression exactly.",
      "expression:str", "math")
    r("describe_numbers", D.describe, "Summary statistics for a list of numbers.",
      "numbers:list[float]", "math")
    r("now", D.now_info, "Current date/time information.", "tz_offset_hours:float=0", "datetime")
    r("date_diff", D.date_diff, "Difference between two dates.", "a:str,b:str", "datetime")
    r("date_add", D.date_add, "Shift a date by days/hours.", "a:str,days:int,hours:int", "datetime")
    r("text_stats", D.text_stats, "All surface statistics of a text in one pass.",
      "text:str", "text")
    r("word_count", D.word_count, "Count words.", "text:str", "text")
    r("char_count", D.char_count, "Count characters.", "text:str,include_spaces:bool", "text")
    r("extract_urls", D.extract_urls, "Extract URLs.", "text:str", "text")
    r("extract_hashtags", D.extract_hashtags, "Extract hashtags.", "text:str", "text")
    r("extract_mentions", D.extract_mentions, "Extract @mentions.", "text:str", "text")
    r("sentences", D.sentences, "Split text into sentences.", "text:str", "text")
    r("regex_search", D.regex_search, "Regex search across a list of texts.",
      "pattern:str,texts:list[str]", "search")
    r("keyword_filter", D.keyword_filter, "Exact keyword filter over records.",
      "records:list[dict],field:str,keywords:list[str],mode:str", "search")
    r("dedupe", D.dedupe, "Exact duplicate detection over a list of strings.",
      "items:list[str]", "data")
    r("parse_json", D.parse_json, "Parse a JSON string.", "text:str", "data")
    r("parse_csv", D.parse_csv, "Parse CSV text into records.", "text:str", "data")
    r("read_csv_file", D.read_csv_file, "Read a CSV file into records.",
      "path:str,limit:int", "data")
    r("sort_records", D.sort_records, "Sort records by a field.",
      "records:list[dict],key:str,reverse:bool", "data")
    r("filter_records", D.filter_records, "Filter records by field/op/value.",
      "records:list[dict],field:str,opname:str,value:Any", "data")
    r("aggregate", D.aggregate, "Aggregate a numeric field, optionally grouped.",
      "records:list[dict],field:str,how:str,group_by:str", "data")
    r("file_metadata", D.file_metadata, "Metadata for one file.", "path:str", "fs")
    r("list_dir", D.list_dir, "List a directory.", "path:str,pattern:str,limit:int", "fs")
    r("sql_query", D.sql_query, "Read-only SELECT against the local agent database.",
      "sql:str,params:tuple,limit:int", "sql")
    r("hash_text", D.hash_text, "Hash a string.", "text:str,algo:str", "util")
    r("system_metrics", D.system_metrics, "Live CPU/RAM metrics.", "", "util")


_bootstrap()
