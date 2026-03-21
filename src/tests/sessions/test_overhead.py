"""Tests for overhead cost tracking — fragment persistence and stats.

Tests the `save_overhead` method on `SessionStore` and the
`overhead_stats` / `global_overhead_stats` query functions.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from rbtr.sessions.kinds import FragmentKind
from rbtr.sessions.overhead import (
    CompactionOverhead,
    CompactionTrigger,
    FactExtractionOverhead,
    FactExtractionSource,
)
from rbtr.sessions.store import SessionStore
from rbtr.usage import SessionUsage

SESSION_A = "session-a"
SESSION_B = "session-b"


@pytest.fixture
def store() -> Generator[SessionStore]:
    with SessionStore(":memory:") as s:
        s.set_context(session_id=SESSION_A)
        yield s


# ── save_overhead persists fragments ─────────────────────────────────


def test_compaction_overhead_persisted(store: SessionStore) -> None:
    """An `overhead-compaction` fragment is persisted with correct data."""
    payload = CompactionOverhead(
        trigger=CompactionTrigger.MANUAL,
        old_messages=10,
        kept_messages=4,
        summary_tokens=500,
        model_name="claude/claude-sonnet-4-20250514",
    )
    row_id = store.save_overhead(
        SESSION_A,
        FragmentKind.OVERHEAD_COMPACTION,
        payload,
        input_tokens=1200,
        output_tokens=300,
        cost=0.005,
    )
    assert row_id

    # Verify it appears in overhead_stats.
    oh = store.overhead_stats(SESSION_A)
    assert oh.compaction_count == 1
    assert oh.compaction_input_tokens == 1200
    assert oh.compaction_output_tokens == 300
    assert oh.compaction_cost == pytest.approx(0.005)


def test_extraction_overhead_persisted(store: SessionStore) -> None:
    """An `overhead-fact-extraction` fragment is persisted with correct data."""
    payload = FactExtractionOverhead(
        source=FactExtractionSource.COMPACTION,
        added=2,
        confirmed=1,
        superseded=0,
        model_name="claude/claude-haiku-4-20250514",
        fact_ids=["f1", "f2", "f3"],
    )
    row_id = store.save_overhead(
        SESSION_A,
        FragmentKind.OVERHEAD_FACT_EXTRACTION,
        payload,
        input_tokens=800,
        output_tokens=100,
        cost=0.001,
    )
    assert row_id

    oh = store.overhead_stats(SESSION_A)
    assert oh.fact_extraction_count == 1
    assert oh.fact_extraction_input_tokens == 800
    assert oh.fact_extraction_output_tokens == 100
    assert oh.fact_extraction_cost == pytest.approx(0.001)


# ── Overhead fragments excluded from conversation ────────────────────


def test_load_messages_skips_overhead(store: SessionStore) -> None:
    """`load_messages` does not return overhead fragments."""
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    # Insert a real message.
    store.save_messages(SESSION_A, [ModelRequest(parts=[UserPromptPart(content="hello")])])

    # Insert overhead.
    store.save_overhead(
        SESSION_A,
        FragmentKind.OVERHEAD_COMPACTION,
        CompactionOverhead(
            trigger=CompactionTrigger.MANUAL,
            old_messages=1,
            kept_messages=1,
            summary_tokens=50,
        ),
        input_tokens=100,
        output_tokens=50,
        cost=0.001,
    )

    messages = store.load_messages(SESSION_A)
    assert len(messages) == 1  # Only the real message.


def test_conversation_stats_exclude_overhead(store: SessionStore) -> None:
    """Token stats only count `request-message`/`response-message` fragments."""
    from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
    from pydantic_ai.usage import RequestUsage

    store.save_messages(
        SESSION_A,
        [
            ModelRequest(parts=[UserPromptPart(content="hello")]),
            ModelResponse(
                parts=[TextPart(content="hi")],
                usage=RequestUsage(input_tokens=100, output_tokens=50),
                model_name="test",
            ),
        ],
        cost=0.01,
    )

    store.save_overhead(
        SESSION_A,
        FragmentKind.OVERHEAD_COMPACTION,
        CompactionOverhead(
            trigger=CompactionTrigger.MANUAL,
            old_messages=1,
            kept_messages=1,
            summary_tokens=50,
        ),
        input_tokens=5000,
        output_tokens=2000,
        cost=0.50,
    )

    ts = store.token_stats(SESSION_A)
    # Conversation tokens — no overhead contamination.
    assert ts.total_input_tokens == 100
    assert ts.total_output_tokens == 50
    assert ts.total_cost == pytest.approx(0.01)


# ── Stats accumulation ───────────────────────────────────────────────


def test_multiple_overheads_accumulate(store: SessionStore) -> None:
    """Multiple overhead fragments in a session are summed."""
    for _ in range(3):
        store.save_overhead(
            SESSION_A,
            FragmentKind.OVERHEAD_COMPACTION,
            CompactionOverhead(
                trigger=CompactionTrigger.AUTO_POST_TURN,
                old_messages=5,
                kept_messages=2,
                summary_tokens=100,
            ),
            input_tokens=1000,
            output_tokens=200,
            cost=0.003,
        )

    store.save_overhead(
        SESSION_A,
        FragmentKind.OVERHEAD_FACT_EXTRACTION,
        FactExtractionOverhead(
            source=FactExtractionSource.COMMAND,
            added=1,
        ),
        input_tokens=500,
        output_tokens=100,
        cost=0.001,
    )

    oh = store.overhead_stats(SESSION_A)
    assert oh.compaction_count == 3
    assert oh.compaction_input_tokens == 3000
    assert oh.compaction_cost == pytest.approx(0.009)
    assert oh.fact_extraction_count == 1
    assert oh.fact_extraction_input_tokens == 500
    assert oh.total_cost == pytest.approx(0.010)


def test_global_overhead_stats(store: SessionStore) -> None:
    """Global overhead stats aggregate across sessions."""
    store.save_overhead(
        SESSION_A,
        FragmentKind.OVERHEAD_COMPACTION,
        CompactionOverhead(
            trigger=CompactionTrigger.MANUAL,
            old_messages=10,
            kept_messages=4,
            summary_tokens=500,
        ),
        input_tokens=1000,
        output_tokens=200,
        cost=0.005,
    )

    store.set_context(session_id=SESSION_B)
    store.save_overhead(
        SESSION_B,
        FragmentKind.OVERHEAD_FACT_EXTRACTION,
        FactExtractionOverhead(
            source=FactExtractionSource.POST,
            added=3,
            fact_ids=["a", "b", "c"],
        ),
        input_tokens=800,
        output_tokens=100,
        cost=0.002,
    )

    oh = store.global_overhead_stats()
    assert oh.compaction_count == 1
    assert oh.fact_extraction_count == 1
    assert oh.compaction_input_tokens == 1000
    assert oh.fact_extraction_input_tokens == 800
    assert oh.total_cost == pytest.approx(0.007)


def test_empty_overhead_stats(store: SessionStore) -> None:
    """No overhead fragments returns empty stats."""
    oh = store.overhead_stats(SESSION_A)
    assert not oh.has_overhead
    assert oh.compaction_count == 0
    assert oh.fact_extraction_count == 0
    assert oh.total_cost == 0.0


# ── SessionUsage tracking ────────────────────────────────────────────


def test_record_compaction_accumulates() -> None:
    """Compaction tokens and cost accumulate on `SessionUsage`."""
    usage = SessionUsage()
    usage.record_compaction(input_tokens=100, output_tokens=50, cost=0.01)
    usage.record_compaction(input_tokens=200, output_tokens=100, cost=0.02)

    assert usage.compaction_input_tokens == 300
    assert usage.compaction_output_tokens == 150
    assert usage.compaction_cost == pytest.approx(0.03)
    assert usage.total_cost == pytest.approx(0.03)


def test_record_fact_extraction_accumulates() -> None:
    """Extraction tokens and cost accumulate on `SessionUsage`."""
    usage = SessionUsage()
    usage.record_fact_extraction(input_tokens=100, output_tokens=50, cost=0.01)

    assert usage.fact_extraction_input_tokens == 100
    assert usage.fact_extraction_output_tokens == 50
    assert usage.fact_extraction_cost == pytest.approx(0.01)
    assert usage.total_cost == pytest.approx(0.01)


def test_overhead_included_in_total_cost() -> None:
    """Overhead cost is added to `total_cost` alongside conversation cost."""
    usage = SessionUsage()
    usage.record_run(input_tokens=1000, output_tokens=500, cost=0.10, new_responses=1)
    usage.record_compaction(input_tokens=100, output_tokens=50, cost=0.01)
    usage.record_fact_extraction(input_tokens=80, output_tokens=20, cost=0.005)

    assert usage.total_cost == pytest.approx(0.115)
    # Conversation tokens unchanged.
    assert usage.input_tokens == 1000
    assert usage.output_tokens == 500


def test_reset_clears_overhead() -> None:
    """`reset()` clears all overhead counters."""
    usage = SessionUsage()
    usage.record_compaction(input_tokens=100, output_tokens=50, cost=0.01)
    usage.record_fact_extraction(input_tokens=80, output_tokens=20, cost=0.005)
    usage.reset()

    assert usage.compaction_input_tokens == 0
    assert usage.compaction_cost == 0.0
    assert usage.fact_extraction_input_tokens == 0
    assert usage.fact_extraction_cost == 0.0
    assert usage.total_cost == 0.0


def test_restore_includes_overhead() -> None:
    """`restore()` sets overhead counters and includes them in total_cost."""
    usage = SessionUsage()
    usage.restore(
        turn_count=5,
        response_count=5,
        input_tokens=5000,
        output_tokens=2000,
        cost=0.50,
        compaction_input_tokens=1000,
        compaction_output_tokens=200,
        compaction_cost=0.05,
        fact_extraction_input_tokens=500,
        fact_extraction_output_tokens=100,
        fact_extraction_cost=0.01,
    )

    assert usage.compaction_input_tokens == 1000
    assert usage.compaction_cost == pytest.approx(0.05)
    assert usage.fact_extraction_input_tokens == 500
    assert usage.fact_extraction_cost == pytest.approx(0.01)
    # total_cost = conversation + compaction + extraction.
    assert usage.total_cost == pytest.approx(0.56)


# ── Fragment kind properties ─────────────────────────────────────────


def test_overhead_kinds_are_overhead() -> None:
    """`is_overhead` returns True for overhead kinds."""
    assert FragmentKind.OVERHEAD_COMPACTION.is_overhead
    assert FragmentKind.OVERHEAD_FACT_EXTRACTION.is_overhead


def test_overhead_kinds_are_not_message() -> None:
    """`is_message` returns False for overhead kinds."""
    assert not FragmentKind.OVERHEAD_COMPACTION.is_message
    assert not FragmentKind.OVERHEAD_FACT_EXTRACTION.is_message


def test_message_kinds_are_not_overhead() -> None:
    """`is_overhead` returns False for message kinds."""
    assert not FragmentKind.REQUEST_MESSAGE.is_overhead
    assert not FragmentKind.RESPONSE_MESSAGE.is_overhead


# ── Clarification produces second fragment ───────────────────────────


def test_clarification_produces_second_extraction_fragment(store: SessionStore) -> None:
    """Extraction + clarification retry produces two overhead fragments."""
    from rbtr.llm.memory import _persist_overhead
    from rbtr.state import EngineState

    state = EngineState()
    state.session_id = SESSION_A

    ctx = MagicMock()
    ctx.store = store
    ctx.state = state

    # Main extraction overhead.
    _persist_overhead(
        ctx,  # type: ignore[arg-type]  # MagicMock duck-types LLMContext
        FactExtractionOverhead(
            source=FactExtractionSource.COMPACTION,
            added=1,
            superseded=1,
            model_name="test-model",
            fact_ids=["f1", "f2"],
        ),
        input_tokens=800,
        output_tokens=100,
        cost=0.002,
    )

    # Clarification overhead.
    _persist_overhead(
        ctx,  # type: ignore[arg-type]  # MagicMock duck-types LLMContext
        FactExtractionOverhead(
            source=FactExtractionSource.COMPACTION,
            model_name="test-model",
        ),
        input_tokens=400,
        output_tokens=50,
        cost=0.001,
    )

    oh = store.overhead_stats(SESSION_A)
    assert oh.fact_extraction_count == 2  # Main + clarification.
    assert oh.fact_extraction_input_tokens == 1200  # 800 + 400.
    assert oh.fact_extraction_output_tokens == 150  # 100 + 50.
    assert oh.fact_extraction_cost == pytest.approx(0.003)


def test_no_clarification_single_fragment(store: SessionStore) -> None:
    """Extraction without clarification produces one overhead fragment."""
    from rbtr.llm.memory import _persist_overhead
    from rbtr.state import EngineState

    state = EngineState()
    state.session_id = SESSION_A

    ctx = MagicMock()
    ctx.store = store
    ctx.state = state

    _persist_overhead(
        ctx,  # type: ignore[arg-type]  # MagicMock duck-types LLMContext
        FactExtractionOverhead(
            source=FactExtractionSource.COMMAND,
            added=2,
            model_name="test-model",
        ),
        input_tokens=800,
        output_tokens=100,
        cost=0.002,
    )

    oh = store.overhead_stats(SESSION_A)
    assert oh.fact_extraction_count == 1
    assert oh.fact_extraction_input_tokens == 800
