"""Tests for toolset registration — ordering and membership.

The entire point of the toolset refactor is controlled tool
presentation order.  These tests lock it down.
"""

from __future__ import annotations

import pytest

# Importing agent triggers all side-effect registrations.
from rbtr.llm.agent import agent  # noqa: F401
from rbtr.llm.tools.common import (
    diff_toolset,
    file_toolset,
    index_toolset,
    review_toolset,
    workspace_toolset,
)

# ── Registration order ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("toolset", "expected_order"),
    [
        (
            index_toolset,
            ["search", "changed_symbols", "find_references", "read_symbol", "list_symbols"],
        ),
        (
            file_toolset,
            ["read_file", "list_files", "grep"],
        ),
        (
            diff_toolset,
            ["changed_files", "diff", "commit_log"],
        ),
        (
            review_toolset,
            [
                "get_pr_discussion",
                "add_draft_comment",
                "edit_draft_comment",
                "remove_draft_comment",
                "set_draft_summary",
                "read_draft",
            ],
        ),
        (
            workspace_toolset,
            ["edit", "remember"],
        ),
    ],
    ids=["index", "file", "diff", "review", "workspace"],
)
def test_tool_registration_order(
    toolset: object,
    expected_order: list[str],
) -> None:
    """Tools within each toolset appear in the intended presentation order."""
    actual = list(toolset.tools.keys())  # type: ignore[union-attr]
    assert actual == expected_order


# ── Total tool count ─────────────────────────────────────────────────


def test_total_tool_count() -> None:
    """All 19 tools are registered across the five toolsets."""
    all_tools = (
        list(index_toolset.tools)
        + list(file_toolset.tools)
        + list(diff_toolset.tools)
        + list(review_toolset.tools)
        + list(workspace_toolset.tools)
    )
    assert len(all_tools) == 19
    assert len(set(all_tools)) == 19, "duplicate tool names across toolsets"
