"""Tests for embedding_text()."""

from __future__ import annotations

from rbtr_legacy.index.embeddings import embedding_text


def test_embedding_text_is_name_plus_content() -> None:
    """Current format: name + newline + content."""
    content = """\
def save_draft(pr: int) -> None:
    \"\"\"Persist draft to disk.\"\"\"
    pass"""
    result = embedding_text("save_draft", content)
    assert result == f"save_draft\n{content}"


def test_embedding_text_name_on_first_line() -> None:
    """Name appears on the first line, content follows."""
    content = """\
def helper():
    return 42"""
    result = embedding_text("helper", content)
    lines = result.split("\n")
    assert lines[0] == "helper"
    assert "def helper" in lines[1]
