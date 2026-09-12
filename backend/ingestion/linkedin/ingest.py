"""LinkedIn export ingestion.

Deterministic end to end: no LLM is involved. Files are identified by their CSV
header signature, not by filename, so a renamed or differently-versioned export
still works. Everything stays on this machine.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from backend.config import LINKEDIN_SOURCES
from backend.db import connect, execute, query
from backend.tools.deterministic import parse_date

csv.field_size_limit(10_000_000)

# doc_type <- required header columns. First match wins; order matters.
HEADER_SIGNATURES: list[tuple[str, set[str]]] = [
    ("post", {"Date", "ShareCommentary"}),
    ("comment", {"Date", "Link", "Message"}),
    ("article", {"Title", "Content"}),
    ("reaction", {"Date", "Type", "Link"}),
    ("vote", {"Date", "Link", "OptionText"}),
    ("repost", {"Date", "Link"}),
    ("message", {"CONVERSATION ID", "FROM", "CONTENT"}),
    ("profile", {"First Name", "Last Name", "Headline"}),
    ("position", {"Company Name", "Title", "Started On"}),
    ("project", {"Title", "Description", "Url"}),
    ("education", {"School Name", "Degree Name"}),
    ("skill", {"Name"}),
    ("certification", {"Name", "Authority"}),
]

# Types that carry the user's own long-form writing (the style corpus).
WRITING_TYPES = ("post", "article", "comment")


@dataclass
class IngestReport:
    sources: list[str] = field(default_factory=list)
    files_seen: int = 0
    files_used: int = 0
    rows_read: int = 0
    inserted: int = 0
    duplicates: int = 0
    skipped_empty: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    unmatched_files: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["duration_s"] = round(self.duration_s, 2)
        return d


def discover_sources(paths: list[str] | None = None) -> list[Path]:
    """Return existing export locations (extracted folders and/or .zip archives)."""
    out: list[Path] = []
    for p in paths if paths is not None else LINKEDIN_SOURCES:
        pp = Path(p)
        if pp.exists() and (pp.is_dir() or zipfile.is_zipfile(pp)):
            out.append(pp)
    return out


def _iter_csv_members(src: Path) -> Iterator[tuple[str, str]]:
    """Yield (member_name, csv_text) for every CSV in a folder or zip archive."""
    if src.is_dir():
        for p in sorted(src.rglob("*.csv")):
            try:
                yield str(p.relative_to(src)), p.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                continue
    else:
        with zipfile.ZipFile(src) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".csv"):
                    continue
                try:
                    yield info.filename, z.read(info).decode("utf-8-sig", errors="replace")
                except (KeyError, zipfile.BadZipFile):
                    continue


def classify_file(headers: list[str]) -> str | None:
    hs = {h.strip() for h in headers}
    for doc_type, required in HEADER_SIGNATURES:
        if required <= hs:
            return doc_type
    return None


def unwrap_quoted_lines(text: str) -> str:
    """LinkedIn's share export wraps every line of a post in double quotes.

    Only triggered when the tell-tale `"\\n"` marker is present, so genuine quoted
    text in normal posts is left alone.
    """
    if '"\n"' not in text:
        return text
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith('"'):
            s = s[1:]
        if s.endswith('"'):
            s = s[:-1]
        out.append(s)
    return "\n".join(out)


def repair_mojibake(text: str) -> str:
    """LinkedIn's export replaces smart punctuation with U+FFFD. Restore it.

    `let\\ufffds` -> apostrophe, `AI\\ufffdthey` -> em dash, stray ones dropped.
    """
    if "�" not in text:
        return text
    t = re.sub(r"(?<=\w)�(?=(?:s|t|re|ll|ve|d|m)\b)", "'", text)
    t = re.sub(r"(?<=\w)�(?=\w)", "—", t)
    t = re.sub(r"\s*�\s*", " ", t)
    return t


def _clean(text: str) -> str:
    """Normalise whitespace but preserve paragraph breaks (they are style signal)."""
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = unwrap_quoted_lines(t)
    t = repair_mojibake(t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _uid(doc_type: str, text: str, ts: str | None) -> str:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    key = "|".join((doc_type, ts or "", norm))
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()


def _iso(value: str | None) -> str | None:
    if not value:
        return None
    d = parse_date(str(value).strip())
    return d.isoformat(timespec="seconds") if d else None


def _owner_name(rows_by_type: dict[str, list[dict]]) -> str | None:
    for r in rows_by_type.get("profile", []):
        first, last = (r.get("First Name") or "").strip(), (r.get("Last Name") or "").strip()
        if first or last:
            return f"{first} {last}".strip()
    return None


def _normalise_row(doc_type: str, row: dict, source_file: str, owner: str | None) -> dict | None:
    """Map one raw CSV row to a document record, or None if it carries no text."""
    text, url, ts, meta = "", None, None, {}

    if doc_type == "post":
        text = row.get("ShareCommentary") or ""
        url = row.get("ShareLink")
        ts = row.get("Date")
        meta = {
            "visibility": row.get("Visibility"),
            "shared_url": row.get("SharedUrl") or None,
            "media_url": row.get("MediaUrl") or None,
            "is_reshare": bool((row.get("SharedUrl") or "").strip()),
        }
    elif doc_type == "comment":
        text, url, ts = row.get("Message") or "", row.get("Link"), row.get("Date")
    elif doc_type == "article":
        text = row.get("Content") or ""
        url, ts = row.get("Url"), row.get("Date") or row.get("PublishedDate")
        meta = {"title": row.get("Title")}
    elif doc_type == "message":
        sender = (row.get("FROM") or "").strip()
        # Only the user's own messages; other people's private text is never stored.
        if not owner or sender.lower() != owner.lower():
            return None
        text, ts = row.get("CONTENT") or "", row.get("DATE")
        meta = {"subject": row.get("SUBJECT") or None, "folder": row.get("FOLDER")}
    elif doc_type == "profile":
        text = " ".join(
            x for x in (row.get("Headline"), row.get("Summary")) if (x or "").strip()
        )
        meta = {k: v for k, v in row.items() if k in ("Industry", "Geo Location", "Websites")}
    elif doc_type == "position":
        text = row.get("Description") or ""
        ts = row.get("Started On")
        meta = {"company": row.get("Company Name"), "title": row.get("Title"),
                "finished_on": row.get("Finished On")}
        if not text.strip():
            text = f"{row.get('Title', '')} at {row.get('Company Name', '')}".strip()
    elif doc_type == "project":
        text = row.get("Description") or ""
        url, ts = row.get("Url"), row.get("Started On")
        meta = {"title": row.get("Title")}
    elif doc_type == "education":
        text = " ".join(x for x in (row.get("Degree Name"), row.get("Notes"),
                                    row.get("Activities")) if (x or "").strip())
        meta = {"school": row.get("School Name")}
    elif doc_type == "skill":
        text = row.get("Name") or ""
    elif doc_type == "certification":
        text = f"{row.get('Name', '')} ({row.get('Authority', '')})"
    elif doc_type in ("reaction", "vote", "repost"):
        # No authored text - kept as engagement/interest signal only.
        text = row.get("OptionText") or ""
        url, ts = row.get("Link"), row.get("Date")
        meta = {"reaction_type": row.get("Type")} if doc_type == "reaction" else {}
        if doc_type == "reaction" and not text:
            text = f"[{row.get('Type', 'REACTION')}]"
        elif doc_type == "repost" and not text:
            text = "[REPOST]"

    text = _clean(text)
    if not text:
        return None
    iso = _iso(ts)
    return {
        "doc_uid": _uid(doc_type, text, iso),
        "doc_type": doc_type,
        "source_file": source_file,
        "created_at": iso,
        "text": text,
        "url": (url or None),
        "metadata_json": json.dumps({k: v for k, v in meta.items() if v not in (None, "")},
                                    ensure_ascii=False),
    }


def ingest(paths: list[str] | None = None, reset: bool = False) -> IngestReport:
    """Parse every discovered export into `linkedin_documents`. Idempotent."""
    t0 = time.perf_counter()
    rep = IngestReport()
    conn = connect()
    if reset:
        execute("DELETE FROM linkedin_documents")

    sources = discover_sources(paths)
    rep.sources = [str(s) for s in sources]

    staged: dict[str, dict] = {}
    raw_by_type: dict[str, list[dict]] = {}
    file_types: list[tuple[str, str, list[dict]]] = []  # (source, doc_type, rows)

    for src in sources:
        for member, text in _iter_csv_members(src):
            rep.files_seen += 1
            reader = csv.DictReader(io.StringIO(text))
            headers = reader.fieldnames or []
            doc_type = classify_file(headers)
            if not doc_type:
                rep.unmatched_files.append(f"{src.name}/{member}")
                continue
            rows = list(reader)
            rep.rows_read += len(rows)
            rep.files_used += 1
            raw_by_type.setdefault(doc_type, []).extend(rows)
            file_types.append((f"{src.name}/{member}", doc_type, rows))

    owner = _owner_name(raw_by_type)

    for source_file, doc_type, rows in file_types:
        for row in rows:
            rec = _normalise_row(doc_type, row, source_file, owner)
            if rec is None:
                rep.skipped_empty += 1
                continue
            if rec["doc_uid"] in staged:
                rep.duplicates += 1
                continue
            staged[rec["doc_uid"]] = rec

    now = time.time()
    before = query("SELECT COUNT(*) c FROM linkedin_documents")[0]["c"]
    conn.executemany(
        "INSERT OR IGNORE INTO linkedin_documents"
        "(doc_uid,doc_type,source_file,created_at,text,url,metadata_json,ingested_at)"
        " VALUES(:doc_uid,:doc_type,:source_file,:created_at,:text,:url,:metadata_json,:ts)",
        [{**r, "ts": now} for r in staged.values()],
    )
    conn.commit()
    after = query("SELECT COUNT(*) c FROM linkedin_documents")[0]["c"]
    rep.inserted = after - before
    rep.duplicates += len(staged) - rep.inserted

    rep.by_type = {
        r["doc_type"]: r["c"]
        for r in query("SELECT doc_type, COUNT(*) c FROM linkedin_documents GROUP BY doc_type")
    }
    if owner:
        from backend.db import remember

        remember("profile", "owner_name", owner, source="ingestion", confidence=1.0)
    rep.duration_s = time.perf_counter() - t0
    return rep


def writing_corpus(limit: int | None = None, types: tuple[str, ...] = WRITING_TYPES,
                   min_words: int = 20) -> list[dict]:
    """The user's own long-form writing, newest first. Style/RAG source of truth."""
    ph = ",".join("?" * len(types))
    sql = (
        f"SELECT id, doc_type, created_at, text, url, metadata_json FROM linkedin_documents "
        f"WHERE doc_type IN ({ph}) AND length(text) >= ? ORDER BY created_at DESC"
    )
    params: list = [*types, min_words * 4]
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = [dict(r) for r in query(sql, params)]
    out = []
    for r in rows:
        if len(r["text"].split()) >= min_words:
            r["metadata"] = json.loads(r.pop("metadata_json") or "{}")
            out.append(r)
    return out
