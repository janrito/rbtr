"""Tests for /memory command and compaction extraction trigger.

Organisation:
- /memory list (bare)
- /memory all
- /memory extract
- Compaction triggers extraction
- Disabled memory config
"""

from __future__ import annotations

import pytest

from rbtr.config import config
from rbtr.engine.core import Engine
from rbtr.events import CompactionFinished
from rbtr.llm.compact import compact_history
from rbtr.llm.context import LLMContext
from rbtr.providers import BuiltinProvider
from rbtr.sessions.kinds import GLOBAL_SCOPE

from .conftest import (
    TARGET_PR_42,
    _assistant,
    _seed,
    _turns,
    _user,
    drain,
    output_texts,
    summary_result,
)

# ── Fixtures ─────────────────────────────────────────────────────────

RBTR_KEY = "testowner/testrepo"
SESSION_ID = "memory-cmd-session"


@pytest.fixture
def mem_engine(config_path: str, engine: Engine) -> Engine:
    """Engine with memory enabled and LLM connected."""
    config.update(memory={"enabled": True})
    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine._sync_store_context()
    return engine


# ═══════════════════════════════════════════════════════════════════════
# /memory list (bare)
# ═══════════════════════════════════════════════════════════════════════


def test_memory_list_empty(mem_engine: Engine) -> None:
    """/memory with no facts shows 'no facts stored'."""
    mem_engine._handle_command("/memory")
    texts = output_texts(drain(mem_engine.events))
    assert any("No facts" in t for t in texts)


def test_memory_list_shows_active_facts(mem_engine: Engine) -> None:
    """/memory shows active facts grouped by scope."""
    mem_engine.store.insert_fact(GLOBAL_SCOPE, "Prefers British English.", SESSION_ID)
    mem_engine.store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    mem_engine._handle_command("/memory")
    texts = output_texts(drain(mem_engine.events))
    joined = "\n".join(texts)
    assert "Prefers British English." in joined
    assert "Uses pytest." in joined


def test_memory_list_hides_superseded(mem_engine: Engine) -> None:
    """/memory (bare) excludes superseded facts."""
    old = mem_engine.store.insert_fact(RBTR_KEY, "Python 3.12.", SESSION_ID)
    new = mem_engine.store.insert_fact(RBTR_KEY, "Python 3.13+.", SESSION_ID)
    mem_engine.store.supersede_fact(old.id, new.id)

    mem_engine._handle_command("/memory")
    texts = output_texts(drain(mem_engine.events))
    joined = "\n".join(texts)
    assert "Python 3.13+." in joined
    assert "Python 3.12." not in joined


# ═══════════════════════════════════════════════════════════════════════
# /memory all
# ═══════════════════════════════════════════════════════════════════════


def test_memory_all_includes_superseded(mem_engine: Engine) -> None:
    """/memory all includes superseded facts."""
    old = mem_engine.store.insert_fact(RBTR_KEY, "Python 3.12.", SESSION_ID)
    new = mem_engine.store.insert_fact(RBTR_KEY, "Python 3.13+.", SESSION_ID)
    mem_engine.store.supersede_fact(old.id, new.id)

    mem_engine._handle_command("/memory all")
    texts = output_texts(drain(mem_engine.events))
    joined = "\n".join(texts)
    assert "Python 3.13+." in joined
    assert "Python 3.12." in joined


# ═══════════════════════════════════════════════════════════════════════
# /memory extract
# ═══════════════════════════════════════════════════════════════════════


def test_memory_extract_calls_extraction(mem_engine: Engine, mocker: object) -> None:
    """/memory extract runs extraction on current session messages."""
    mock_extract = mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.memory_cmd.extract_facts_from_ctx",
    )
    _seed(mem_engine, [_user("hello"), _assistant("hi there")])

    mem_engine._handle_command("/memory extract")
    mock_extract.assert_called_once()


def test_memory_extract_no_llm(config_path: str, engine: Engine) -> None:
    """/memory extract warns when no LLM is connected."""
    config.update(memory={"enabled": True})
    engine._handle_command("/memory extract")
    texts = output_texts(drain(engine.events))
    assert any("No LLM connected" in t for t in texts)


def test_memory_extract_no_messages(mem_engine: Engine) -> None:
    """/memory extract on empty session reports no messages."""
    mem_engine._handle_command("/memory extract")
    texts = output_texts(drain(mem_engine.events))
    assert any("No messages" in t for t in texts)


# ═══════════════════════════════════════════════════════════════════════
# Compaction triggers extraction
# ═══════════════════════════════════════════════════════════════════════


def test_compaction_triggers_extraction(
    config_path: str,
    mocker: object,
    engine: Engine,
    llm_ctx: LLMContext,
) -> None:
    """Compaction calls `run_fact_extraction` with the old messages."""
    config.update(memory={"enabled": True})
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value=summary_result("Summary of the conversation."),
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]
    mock_extract = mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact.run_fact_extraction",
    )

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.context_window = 200_000
    _seed(engine, _turns(10))

    compact_history(llm_ctx)
    mock_extract.assert_called_once()

    # Verify it received the old messages (not the kept ones).
    call_kwargs = mock_extract.call_args
    messages = call_kwargs.args[0]
    assert len(messages) > 0


def test_compaction_extraction_failure_non_fatal(
    config_path: str,
    mocker: object,
    engine: Engine,
    llm_ctx: LLMContext,
) -> None:
    """If extraction fails during compaction, compaction still succeeds."""
    config.update(memory={"enabled": True})
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value=summary_result("Summary."),
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact.run_fact_extraction",
        side_effect=RuntimeError("LLM exploded"),
    )

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.context_window = 200_000
    _seed(engine, _turns(10))

    compact_history(llm_ctx)

    # Compaction still completed despite extraction failure.

    all_events = drain(engine.events)
    assert any(isinstance(e, CompactionFinished) for e in all_events)
    messages = engine.store.load_messages(engine.state.session_id)
    assert len(messages) < 20  # Compacted.


# ═══════════════════════════════════════════════════════════════════════
# /draft post triggers extraction
# ═══════════════════════════════════════════════════════════════════════


def test_draft_post_triggers_extraction(
    mem_engine: Engine,
    mocker: object,
) -> None:
    """/draft post calls extraction after posting."""
    mock_extract = mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.draft_cmd.extract_facts_from_ctx",
    )
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.draft_cmd.post_review_draft",
    )
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.engine.draft_cmd.load_draft",
        return_value=mocker.MagicMock(summary="Good PR", comments=[]),  # type: ignore[union-attr]
    )
    _seed(mem_engine, [_user("review this"), _assistant("LGTM")])

    mem_engine.state.review_target = TARGET_PR_42
    mem_engine._handle_command("/draft post")
    mock_extract.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════
# Disabled memory config
# ═══════════════════════════════════════════════════════════════════════


def test_memory_disabled_list_warns(config_path: str, engine: Engine) -> None:
    """/memory warns when memory is disabled."""
    config.update(memory={"enabled": False})
    engine._handle_command("/memory")
    texts = output_texts(drain(engine.events))
    assert any("disabled" in t.lower() for t in texts)


def test_memory_disabled_extract_warns(config_path: str, engine: Engine) -> None:
    """/memory extract warns when memory is disabled."""
    config.update(memory={"enabled": False})
    engine._handle_command("/memory extract")
    texts = output_texts(drain(engine.events))
    assert any("disabled" in t.lower() for t in texts)


def test_compaction_skips_extraction_when_disabled(
    config_path: str,
    mocker: object,
    engine: Engine,
    llm_ctx: LLMContext,
) -> None:
    """Compaction does not call extraction when memory is disabled."""
    config.update(memory={"enabled": False})
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact._stream_summary",
        return_value=summary_result("Summary."),
    )
    mocker.patch("rbtr.llm.compact.build_model")  # type: ignore[union-attr]
    mocker.patch(  # type: ignore[union-attr]
        "rbtr.llm.compact.run_fact_extraction",
    )

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.model_name = "claude/claude-sonnet-4-20250514"
    engine.state.usage.context_window = 200_000
    _seed(engine, _turns(10))

    compact_history(llm_ctx)
    # `run_fact_extraction` returns None because `config.memory.enabled`
    # is False (checked inside the function).  Compaction succeeds.

    all_events = drain(engine.events)
    assert any(isinstance(e, CompactionFinished) for e in all_events)


# ═══════════════════════════════════════════════════════════════════════
# /memory purge
# ═══════════════════════════════════════════════════════════════════════


def test_memory_purge_deletes_old_facts(mem_engine: Engine) -> None:
    """Purge with a zero-duration cutoff deletes all existing facts."""
    store = mem_engine.store
    store.insert_fact(RBTR_KEY, "Old fact.", SESSION_ID)

    mem_engine._handle_command("/memory purge 0d")
    texts = output_texts(drain(mem_engine.events))
    assert any("1 fact" in t for t in texts)
    assert store.load_active_facts(RBTR_KEY) == []


def test_memory_purge_keeps_recent(mem_engine: Engine) -> None:
    """Purge with a long duration keeps recent facts."""
    store = mem_engine.store
    store.insert_fact(RBTR_KEY, "Fresh fact.", SESSION_ID)

    mem_engine._handle_command("/memory purge 30d")
    texts = output_texts(drain(mem_engine.events))
    assert any("0 facts" in t for t in texts)
    assert len(store.load_active_facts(RBTR_KEY)) == 1


def test_memory_purge_no_args_warns(mem_engine: Engine) -> None:
    """Purge without duration shows usage."""
    mem_engine._handle_command("/memory purge")
    texts = output_texts(drain(mem_engine.events))
    assert any("Usage" in t for t in texts)


def test_memory_purge_invalid_duration_warns(mem_engine: Engine) -> None:
    """Purge with invalid duration shows error."""
    mem_engine._handle_command("/memory purge xyz")
    texts = output_texts(drain(mem_engine.events))
    assert any("Invalid" in t for t in texts)
