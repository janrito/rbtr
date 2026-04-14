"""Tests for bench_search.py extraction logic."""

from __future__ import annotations

import json
import sqlite3
import sys
import textwrap
from pathlib import Path

import pytest

# Add scripts dir to sys.path so bench_search is importable.
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from bench_search import (  # type: ignore[import-untyped,import-not-found]  # noqa: E402
    _parse_search_result_names,
    extract_events,
)

# ── _parse_search_result_names ───────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            "function add_user_prompt  (path/to/file.py:42)\n"
            "function add_lens_prompt  (path/to/file.py:75)",
            {"add_user_prompt", "add_lens_prompt"},
            id="plain-symbol-results",
        ),
        pytest.param(
            "[19.54] function test_foo  (path/to/file.py:10)\n"
            "[3.87] raw_chunk params.yaml:91-140  (params.yaml:91)",
            {"test_foo", "params.yaml:91-140"},
            id="bm25-score-prefix",
        ),
        pytest.param("", set(), id="empty"),
        pytest.param(
            "class Label  (models.py:5)",
            {"Label"},
            id="single-class",
        ),
    ],
)
def test_parse_search_result_names(content: str, expected: set[str]) -> None:
    assert _parse_search_result_names(content) == expected


# ── extract_events ───────────────────────────────────────────────────


def _make_db(tmp_path: Path, fragments: list[dict]) -> Path:
    """Create a minimal sessions.db with the given fragments."""
    db_path = tmp_path / "sessions.db"
    con = sqlite3.connect(str(db_path))
    con.execute(
        textwrap.dedent("""\
        CREATE TABLE fragments (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT,
            fragment_index INTEGER NOT NULL,
            fragment_kind TEXT NOT NULL,
            created_at TEXT NOT NULL,
            session_label TEXT,
            repo_owner TEXT,
            repo_name TEXT,
            model_name TEXT,
            review_target TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cache_write_tokens INTEGER,
            cost REAL,
            data_json TEXT,
            user_text TEXT,
            tool_name TEXT,
            compacted_by TEXT,
            complete INTEGER NOT NULL DEFAULT 0
        )
        """)
    )
    for f in fragments:
        con.execute(
            """
            INSERT INTO fragments (
                id, session_id, fragment_index, fragment_kind,
                created_at, repo_owner, repo_name, data_json, tool_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f["id"],
                f.get("session_id", "s1"),
                f.get("fragment_index", 0),
                f["fragment_kind"],
                f["created_at"],
                f.get("repo_owner"),
                f.get("repo_name"),
                f.get("data_json"),
                f.get("tool_name"),
            ),
        )
    con.commit()
    con.close()
    return db_path


def _tool_call(
    frag_id: str,
    tool_name: str,
    args: dict,
    created_at: str,
    *,
    session_id: str = "s1",
    fragment_index: int = 0,
    repo_owner: str | None = "org",
    repo_name: str | None = "repo",
) -> dict:
    return {
        "id": frag_id,
        "session_id": session_id,
        "fragment_index": fragment_index,
        "fragment_kind": "tool-call",
        "created_at": created_at,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "data_json": json.dumps({"tool_name": tool_name, "args": json.dumps(args)}),
        "tool_name": tool_name,
    }


def _tool_return(
    frag_id: str,
    tool_name: str,
    content: str,
    created_at: str,
    *,
    session_id: str = "s1",
    fragment_index: int = 0,
    repo_owner: str | None = "org",
    repo_name: str | None = "repo",
) -> dict:
    return {
        "id": frag_id,
        "session_id": session_id,
        "fragment_index": fragment_index,
        "fragment_kind": "tool-return",
        "created_at": created_at,
        "repo_owner": repo_owner,
        "repo_name": repo_name,
        "data_json": json.dumps({"tool_name": tool_name, "content": content}),
        "tool_name": tool_name,
    }


def _user_prompt(
    frag_id: str,
    created_at: str,
    *,
    session_id: str = "s1",
) -> dict:
    return {
        "id": frag_id,
        "session_id": session_id,
        "fragment_index": 0,
        "fragment_kind": "user-prompt",
        "created_at": created_at,
        "repo_owner": None,
        "repo_name": None,
        "data_json": json.dumps({"content": "user message"}),
        "tool_name": None,
    }


def test_search_followed_by_read_symbol(tmp_path: Path) -> None:
    """Search immediately followed by read_symbol pairs correctly."""
    db = _make_db(
        tmp_path,
        [
            _tool_call("f1", "search_symbols", {"name": "Foo"}, "2026-01-01T10:00:00"),
            _tool_return(
                "f2",
                "search_symbols",
                "class Foo  (models.py:5)",
                "2026-01-01T10:00:01",
            ),
            _tool_call(
                "f3",
                "read_symbol",
                {"name": "Foo"},
                "2026-01-01T10:00:02",
            ),
        ],
    )
    events = extract_events(db)
    assert len(events) == 1
    assert events[0].query == "Foo"
    assert events[0].read_target == "Foo"


def test_search_without_read(tmp_path: Path) -> None:
    """Search with no subsequent read_symbol has no target."""
    db = _make_db(
        tmp_path,
        [
            _tool_call("f1", "search_codebase", {"query": "foo"}, "2026-01-01T10:00:00"),
            _tool_return("f2", "search_codebase", "no results", "2026-01-01T10:00:01"),
        ],
    )
    events = extract_events(db)
    assert len(events) == 1
    assert events[0].query == "foo"
    assert events[0].read_target is None


def test_read_symbol_beyond_intervening_limit_skipped(tmp_path: Path) -> None:
    """A read_symbol far from search is not paired unless in results."""
    fragments = [
        _tool_call("f1", "search_symbols", {"name": "Foo"}, "2026-01-01T10:00:00"),
        _tool_return(
            "f1r",
            "search_symbols",
            "class Foo  (models.py:5)",
            "2026-01-01T10:00:01",
        ),
    ]
    # 6 intervening tool-calls.
    for idx in range(6):
        fragments.append(
            _tool_call(
                f"g{idx}",
                "read_file",
                {"path": f"file{idx}.py"},
                f"2026-01-01T10:00:0{idx + 2}",
            )
        )
    # read_symbol for "Unrelated" which was NOT in search results.
    fragments.append(
        _tool_call(
            "f_read",
            "read_symbol",
            {"name": "Unrelated"},
            "2026-01-01T10:00:10",
        )
    )
    db = _make_db(tmp_path, fragments)
    events = extract_events(db)
    assert len(events) == 1
    assert events[0].read_target is None


def test_read_symbol_beyond_limit_paired_if_in_results(tmp_path: Path) -> None:
    """A distant read_symbol IS paired if target was in search results."""
    fragments = [
        _tool_call("f1", "search_symbols", {"name": "Foo"}, "2026-01-01T10:00:00"),
        _tool_return(
            "f1r",
            "search_symbols",
            "class Foo  (models.py:5)\nclass Bar  (models.py:20)",
            "2026-01-01T10:00:01",
        ),
    ]
    for idx in range(6):
        fragments.append(
            _tool_call(
                f"g{idx}",
                "read_file",
                {"path": f"file{idx}.py"},
                f"2026-01-01T10:00:0{idx + 2}",
            )
        )
    # read_symbol for Bar which WAS in results.
    fragments.append(
        _tool_call(
            "f_read",
            "read_symbol",
            {"name": "Bar"},
            "2026-01-01T10:00:10",
        )
    )
    db = _make_db(tmp_path, fragments)
    events = extract_events(db)
    assert len(events) == 1
    assert events[0].read_target == "Bar"


def test_user_prompt_stops_pairing(tmp_path: Path) -> None:
    """A user-prompt boundary prevents pairing."""
    db = _make_db(
        tmp_path,
        [
            _tool_call("f1", "search_symbols", {"name": "Foo"}, "2026-01-01T10:00:00"),
            _user_prompt("u1", "2026-01-01T10:00:05"),
            _tool_call(
                "f2",
                "read_symbol",
                {"name": "Foo"},
                "2026-01-01T10:00:10",
            ),
        ],
    )
    events = extract_events(db)
    assert len(events) == 1
    assert events[0].read_target is None


def test_normalises_search_codebase_query(tmp_path: Path) -> None:
    """search_codebase uses 'query' arg, not 'name'."""
    db = _make_db(
        tmp_path,
        [
            _tool_call(
                "f1",
                "search_codebase",
                {"query": "some pattern"},
                "2026-01-01T10:00:00",
            ),
        ],
    )
    events = extract_events(db)
    assert len(events) == 1
    assert events[0].query == "some pattern"
    assert events[0].original_tool == "search_codebase"


def test_multiple_sessions_kept_separate(tmp_path: Path) -> None:
    """Events from different sessions don't cross-contaminate."""
    db = _make_db(
        tmp_path,
        [
            _tool_call(
                "f1",
                "search_symbols",
                {"name": "Foo"},
                "2026-01-01T10:00:00",
                session_id="s1",
            ),
            # read_symbol in different session should not pair.
            _tool_call(
                "f2",
                "read_symbol",
                {"name": "Foo"},
                "2026-01-01T10:00:01",
                session_id="s2",
            ),
        ],
    )
    events = extract_events(db)
    assert len(events) == 1
    assert events[0].read_target is None


def test_retry_chain_detected(tmp_path: Path) -> None:
    """Consecutive searches with overlapping terms are marked as retries."""
    db = _make_db(
        tmp_path,
        [
            _tool_call("f1", "search_symbols", {"name": "Label"}, "2026-01-01T10:00:00"),
            _tool_call(
                "f2",
                "search_codebase",
                {"query": "class Label("},
                "2026-01-01T10:00:05",
            ),
        ],
    )
    events = extract_events(db)
    assert len(events) == 2
    assert not events[0].is_retry
    assert events[1].is_retry
    assert events[1].first_query_in_chain == "Label"


def test_no_retry_across_sessions(tmp_path: Path) -> None:
    """Retry detection doesn't cross session boundaries."""
    db = _make_db(
        tmp_path,
        [
            _tool_call(
                "f1",
                "search_symbols",
                {"name": "Label"},
                "2026-01-01T10:00:00",
                session_id="s1",
            ),
            _tool_call(
                "f2",
                "search_codebase",
                {"query": "class Label("},
                "2026-01-01T10:00:05",
                session_id="s2",
            ),
        ],
    )
    events = extract_events(db)
    assert len(events) == 2
    assert not events[0].is_retry
    assert not events[1].is_retry
