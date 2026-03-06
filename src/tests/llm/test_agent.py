"""Tests for agent instructions — index status prompt injection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from rbtr.llm.agent import AgentDeps, _index_status as index_status
from rbtr.llm.tools.common import _index_tool_names
from rbtr.models import BranchTarget
from rbtr.state import EngineState


@dataclass
class _FakeCtx:
    """Minimal RunContext substitute for testing instructions."""

    deps: AgentDeps


_NOW = datetime.now(tz=UTC)


def _make_ctx(*, review_target=None, index_ready: bool = False) -> _FakeCtx:
    state = EngineState()
    state.review_target = review_target
    state.index_ready = index_ready
    return _FakeCtx(deps=AgentDeps(state=state))


def test_no_review_target_empty() -> None:
    """When no review target is set, index status is empty."""
    ctx = _make_ctx()
    result = index_status(ctx)  # type: ignore[arg-type]
    assert result == ""


def test_index_ready_mentions_tools() -> None:
    """When the index is ready, the prompt encourages tool use."""
    target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=_NOW,
    )
    ctx = _make_ctx(review_target=target, index_ready=True)
    result = index_status(ctx)  # type: ignore[arg-type]
    assert "ready" in result.lower()
    assert "search" in result


def test_index_not_ready_warns() -> None:
    """When the index is still building, the prompt warns about unavailable tools."""
    target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=_NOW,
    )
    ctx = _make_ctx(review_target=target, index_ready=False)
    result = index_status(ctx)  # type: ignore[arg-type]
    assert "still building" in result.lower()
    assert "wait" in result.lower()
    assert "search" in result
    assert "changed_symbols" in result


def test_index_tool_names_matches_registered_tools() -> None:
    """Introspected tool list includes all tools guarded by _require_index."""
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
    target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=_NOW,
    )
    names = _index_tool_names()

    ready = index_status(_make_ctx(review_target=target, index_ready=True))  # type: ignore[arg-type]
    not_ready = index_status(_make_ctx(review_target=target, index_ready=False))  # type: ignore[arg-type]
    for name in names:
        assert name in ready, f"{name} missing from ready prompt"
        assert name in not_ready, f"{name} missing from not-ready prompt"
