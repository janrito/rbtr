"""Tests for agent instructions — index status prompt injection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from rbtr.engine.agent import AgentDeps, _index_tool_names, index_status
from rbtr.engine.session import Session
from rbtr.models import BranchTarget


@dataclass
class _FakeCtx:
    """Minimal RunContext substitute for testing instructions."""

    deps: AgentDeps


_NOW = datetime.now(tz=UTC)


def _make_ctx(*, review_target=None, index_ready: bool = False) -> _FakeCtx:
    session = Session()
    session.review_target = review_target
    session.index_ready = index_ready
    return _FakeCtx(deps=AgentDeps(session=session))


def test_no_review_target_empty() -> None:
    """When no review target is set, index status is empty."""
    ctx = _make_ctx()
    result = index_status(ctx)  # type: ignore[arg-type]
    assert result == ""


def test_index_ready_mentions_tools() -> None:
    """When the index is ready, the prompt encourages tool use."""
    target = BranchTarget(base_branch="main", head_branch="feature", updated_at=_NOW)
    ctx = _make_ctx(review_target=target, index_ready=True)
    result = index_status(ctx)  # type: ignore[arg-type]
    assert "ready" in result.lower()
    assert "search_symbols" in result
    assert "search_codebase" in result
    assert "search_similar" in result


def test_index_not_ready_warns() -> None:
    """When the index is still building, the prompt warns about unavailable tools."""
    target = BranchTarget(base_branch="main", head_branch="feature", updated_at=_NOW)
    ctx = _make_ctx(review_target=target, index_ready=False)
    result = index_status(ctx)  # type: ignore[arg-type]
    assert "still building" in result.lower()
    assert "search_symbols" in result
    assert "semantic_diff" in result


def test_index_tool_names_matches_registered_tools() -> None:
    """Introspected tool list includes all tools guarded by _require_index."""
    names = _index_tool_names()
    assert len(names) >= 5, f"expected ≥5 index tools, got {names}"
    assert "search_symbols" in names
    assert "read_symbol" in names
    assert "semantic_diff" in names
    # Git-only tools should NOT appear.
    assert "diff" not in names
    assert "commit_log" not in names


def test_all_index_tools_appear_in_prompt() -> None:
    """Every introspected index tool name appears in both ready and not-ready prompts."""
    target = BranchTarget(base_branch="main", head_branch="feature", updated_at=_NOW)
    names = _index_tool_names()

    ready = index_status(_make_ctx(review_target=target, index_ready=True))  # type: ignore[arg-type]
    not_ready = index_status(_make_ctx(review_target=target, index_ready=False))  # type: ignore[arg-type]
    for name in names:
        assert name in ready, f"{name} missing from ready prompt"
        assert name in not_ready, f"{name} missing from not-ready prompt"
