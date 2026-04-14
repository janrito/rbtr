"""Tests for agent instructions — index status prompt injection."""

from __future__ import annotations

from rbtr_legacy.llm.tools.common import _index_tool_names
from rbtr_legacy.prompts import render_index_status


def test_index_ready_mentions_tools() -> None:
    """When the index is ready, the prompt encourages tool use."""
    result = render_index_status(status="ready", tool_names=_index_tool_names())
    assert "ready" in result.lower()
    assert "search" in result


def test_index_not_ready_warns() -> None:
    """When the index is still building, the prompt warns about unavailable tools."""
    result = render_index_status(status="building", tool_names=_index_tool_names())
    assert "still building" in result.lower()
    assert "wait" in result.lower()
    assert "search" in result
    assert "changed_symbols" in result


def test_index_tool_names_matches_registered_tools() -> None:
    """Introspected tool list includes all tools registered on index_toolset."""
    names = _index_tool_names()
    assert len(names) >= 5, f"expected ≥5 index tools, got {names}"
    assert "search" in names
    assert "read_symbol" in names
    assert "changed_symbols" in names
    # Git-only tools should NOT appear.
    assert "diff" not in names
    assert "commit_log" not in names


def test_all_index_tools_appear_in_prompt() -> None:
    """Every introspected index tool name appears in both ready and not-ready prompts."""
    names = _index_tool_names()

    ready = render_index_status(status="ready", tool_names=names)
    not_ready = render_index_status(status="building", tool_names=names)
    for name in names:
        assert name in ready, f"{name} missing from ready prompt"
        assert name in not_ready, f"{name} missing from not-ready prompt"
