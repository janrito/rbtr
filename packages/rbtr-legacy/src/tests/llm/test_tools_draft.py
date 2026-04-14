"""Tests for draft tools — find_comment."""

from __future__ import annotations

import pytest

from rbtr_legacy.github.draft import find_comment
from rbtr_legacy.models import InlineComment

# ── find_comment ─────────────────────────────────────────────────────

_COMMENTS = [
    InlineComment(path="a.py", line=10, body="**blocker:** Bug here."),
    InlineComment(path="a.py", line=20, body="**nit:** Rename this."),
    InlineComment(path="b.py", line=5, body="**blocker:** Missing check."),
]


class TestFindComment:
    """Tests for find_comment — locate a comment by path + body substring."""

    @pytest.mark.parametrize(
        ("path", "substr", "expected_index", "expected_line"),
        [
            ("a.py", "Bug here", 0, 10),
            ("a.py", "Rename", 1, 20),
            ("b.py", "Missing", 2, 5),
        ],
    )
    def test_found(self, path: str, substr: str, expected_index: int, expected_line: int) -> None:
        result = find_comment(_COMMENTS, path, substr)
        assert isinstance(result, tuple)
        idx, comment = result
        assert idx == expected_index
        assert comment.line == expected_line

    @pytest.mark.parametrize(
        ("path", "substr", "error_fragment"),
        [
            ("a.py", "nonexistent", "No comment"),
            ("c.py", "Bug", "No comments on"),
        ],
    )
    def test_not_found(self, path: str, substr: str, error_fragment: str) -> None:
        result = find_comment(_COMMENTS, path, substr)
        assert isinstance(result, str)
        assert error_fragment in result

    def test_ambiguous(self) -> None:
        result = find_comment(_COMMENTS, "a.py", "**")
        assert isinstance(result, str)
        assert "2 comments" in result

    def test_empty_comments(self) -> None:
        result = find_comment([], "a.py", "anything")
        assert isinstance(result, str)
        assert "No comments" in result

    def test_wrong_path_right_body(self) -> None:
        """Substring matches body but path doesn't — no result."""
        result = find_comment(_COMMENTS, "b.py", "Bug here")
        assert isinstance(result, str)
        assert "No comment" in result
