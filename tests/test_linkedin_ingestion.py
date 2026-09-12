import csv
import io

from backend.ingestion.linkedin import ingest as ING


def test_classify_file_by_header_not_filename():
    assert ING.classify_file(["Date", "ShareLink", "ShareCommentary", "Visibility"]) == "post"
    assert ING.classify_file(["Date", "Link", "Message"]) == "comment"
    assert ING.classify_file(["Date", "Type", "Link"]) == "reaction"
    assert ING.classify_file(["First Name", "Last Name", "Headline"]) == "profile"
    assert ING.classify_file(["Company Name", "Title", "Started On"]) == "position"
    assert ING.classify_file(["Totally", "Unknown", "Columns"]) is None


def test_unwrap_quoted_lines_only_when_marker_present():
    wrapped = 'First line."\n"Second line."\n"Third line.'
    out = ING.unwrap_quoted_lines(wrapped)
    assert out == "First line.\nSecond line.\nThird line."
    normal = 'He said "this is fine" and moved on.'
    assert ING.unwrap_quoted_lines(normal) == normal


def test_repair_mojibake():
    assert ING.repair_mojibake("let�s connect") == "let's connect"
    assert ING.repair_mojibake("AI�they just need tools") == "AI—they just need tools"
    assert "�" not in ING.repair_mojibake("odd � spacing")


def test_normalise_post_row():
    row = {"Date": "2026-08-22 16:57:52", "ShareLink": "https://x/1",
           "ShareCommentary": "Body text here", "SharedUrl": "", "MediaUrl": "",
           "Visibility": "MEMBER_NETWORK"}
    rec = ING._normalise_row("post", row, "Shares.csv", "jimit talekar")
    assert rec["doc_type"] == "post"
    assert rec["created_at"].startswith("2026-08-22T16:57:52")
    assert rec["text"] == "Body text here"
    assert len(rec["doc_uid"]) == 40


def test_empty_rows_are_skipped():
    assert ING._normalise_row("post", {"Date": "2026-01-01", "ShareCommentary": "  "},
                              "f.csv", None) is None


def test_only_own_messages_are_kept():
    own = {"CONVERSATION ID": "1", "FROM": "Jimit Talekar", "CONTENT": "hello there",
           "DATE": "2026-01-01 10:00:00"}
    other = {**own, "FROM": "Someone Else", "CONTENT": "private reply"}
    assert ING._normalise_row("message", own, "messages.csv", "jimit talekar") is not None
    assert ING._normalise_row("message", other, "messages.csv", "jimit talekar") is None
    # Without a known owner, no private message is stored at all.
    assert ING._normalise_row("message", own, "messages.csv", None) is None


def test_uid_is_stable_and_distinguishing():
    a = ING._uid("post", "Same text", "2026-01-01T00:00:00")
    b = ING._uid("post", "same   TEXT", "2026-01-01T00:00:00")
    c = ING._uid("post", "Different", "2026-01-01T00:00:00")
    assert a == b and a != c


def test_csv_parsing_roundtrip_matches_source_rows():
    sample = 'Date,Link,Message\n2026-01-01 00:00:00,https://x,"Hello, world"\n'
    rows = list(csv.DictReader(io.StringIO(sample)))
    assert ING.classify_file(list(rows[0].keys())) == "comment"
    rec = ING._normalise_row("comment", rows[0], "Comments.csv", None)
    assert rec["text"] == "Hello, world"


def test_corpus_query_filters_short_documents():
    docs = ING.writing_corpus(min_words=500)
    assert all(len(d["text"].split()) >= 500 for d in docs)
