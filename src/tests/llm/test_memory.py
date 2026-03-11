"""Tests for cross-session memory extraction logic.

Tests the pure processing layer (``process_extracted_facts``) and
prompt rendering without requiring an LLM.  The LLM makes the
dedup decision — these tests verify we correctly apply what the
LLM returns.

Organisation:
- Fixtures
- process_extracted_facts — new facts
- process_extracted_facts — confirm
- process_extracted_facts — supersede
- process_extracted_facts — scope handling
- process_extracted_facts — empty
- Prompt rendering
- FactExtractionResult model
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from pytest_mock import MockerFixture

from rbtr.llm.memory import (
    ExtractedFact,
    FactAction,
    FactExtractionResult,
    _build_clarify_prompt,
    _build_user_prompt,
    _clarify_failed_facts,
    process_extracted_facts,
    render_facts_instruction,
)
from rbtr.sessions.store import SessionStore
from tests.sessions.fact_data import GLOBAL, RBTR_KEY

SESSION_ID = "extract-session-001"


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def store() -> Generator[SessionStore]:
    """Empty in-memory store."""
    s = SessionStore()
    yield s
    s.close()


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — new facts
# ═══════════════════════════════════════════════════════════════════════


def test_new_facts_inserted(store: SessionStore) -> None:
    """New facts are inserted into the store."""
    extracted = [
        ExtractedFact(content="Uses pytest.", scope="repo", action=FactAction.NEW),
        ExtractedFact(content="Python 3.13+.", scope="repo", action=FactAction.NEW),
        ExtractedFact(content="Prefers British English.", scope="global", action=FactAction.NEW),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 3
    assert pr.confirmed == 0
    assert pr.superseded == 0

    repo_facts = store.load_active_facts(RBTR_KEY)
    global_facts = store.load_active_facts(GLOBAL)
    assert len(repo_facts) == 2
    assert len(global_facts) == 1


def test_new_fact_inserted_directly(store: SessionStore) -> None:
    """A fact tagged 'new' by the LLM is inserted without client-side dedup."""
    store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    # LLM says "new" — we trust it and insert.
    extracted = [
        ExtractedFact(content="Uses pytest.", scope="repo", action=FactAction.NEW),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 1
    assert pr.confirmed == 0

    # Two facts now — the LLM should have tagged it 'confirm',
    # but if it didn't, `/memory purge` handles cleanup later.
    facts = store.load_active_facts(RBTR_KEY)
    assert len(facts) == 2


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — confirm
# ═══════════════════════════════════════════════════════════════════════


def test_confirm_bumps_existing(store: SessionStore) -> None:
    """Confirm action bumps an existing fact."""
    store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    extracted = [
        ExtractedFact(
            content="Uses pytest.",
            scope="repo",
            action=FactAction.CONFIRM,
            existing_content="Uses pytest.",
        ),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 0
    assert pr.confirmed == 1

    facts = store.load_active_facts(RBTR_KEY)
    assert len(facts) == 1
    assert facts[0].confirm_count == 2


def test_confirm_content_not_found(store: SessionStore) -> None:
    """Confirm with non-matching content is silently ignored."""
    extracted = [
        ExtractedFact(
            content="Uses pytest.",
            scope="repo",
            action=FactAction.CONFIRM,
            existing_content="No such fact.",
        ),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 0
    assert pr.confirmed == 0


def test_confirm_without_existing_content_falls_through(store: SessionStore) -> None:
    """Confirm without existing_content is treated as new (fallback)."""
    extracted = [
        ExtractedFact(content="Orphan confirm.", scope="repo", action=FactAction.CONFIRM),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 1
    assert pr.confirmed == 0


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — supersede
# ═══════════════════════════════════════════════════════════════════════


def test_supersede_replaces_old(store: SessionStore) -> None:
    """Supersede action inserts new and marks old as superseded."""
    store.insert_fact(RBTR_KEY, "Python 3.12.", SESSION_ID)

    extracted = [
        ExtractedFact(
            content="Python 3.13+.",
            scope="repo",
            action=FactAction.SUPERSEDE,
            existing_content="Python 3.12.",
        ),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 1
    assert pr.superseded == 1

    facts = store.load_active_facts(RBTR_KEY)
    assert len(facts) == 1
    assert facts[0].content == "Python 3.13+."


def test_supersede_content_not_found_skipped(store: SessionStore) -> None:
    """Supersede with non-matching content is skipped entirely."""
    extracted = [
        ExtractedFact(
            content="New replacement.",
            scope="repo",
            action=FactAction.SUPERSEDE,
            existing_content="No such fact.",
        ),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 0
    assert pr.superseded == 0

    facts = store.load_active_facts(RBTR_KEY)
    assert len(facts) == 0


def test_supersede_without_existing_content_falls_through(store: SessionStore) -> None:
    """Supersede without existing_content is treated as new (fallback)."""
    extracted = [
        ExtractedFact(content="Orphan supersede.", scope="repo", action=FactAction.SUPERSEDE),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 1
    assert pr.superseded == 0


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — failed collection
# ═══════════════════════════════════════════════════════════════════════


def test_supersede_mismatch_collected_as_failed(store: SessionStore) -> None:
    """Supersede with non-matching content is collected in `failed`."""
    store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    extracted = [
        ExtractedFact(
            content="Uses pytest with coverage.",
            scope="repo",
            action=FactAction.SUPERSEDE,
            existing_content="Uses pytst.",  # typo
        ),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 0
    assert pr.superseded == 0
    assert len(pr.failed) == 1
    assert pr.failed[0].existing_content == "Uses pytst."


def test_confirm_mismatch_collected_as_failed(store: SessionStore) -> None:
    """Confirm with non-matching content is collected in `failed`."""
    store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    extracted = [
        ExtractedFact(
            content="Uses pytest.",
            scope="repo",
            action=FactAction.CONFIRM,
            existing_content="Uses pytst.",  # typo
        ),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.confirmed == 0
    assert len(pr.failed) == 1


def test_mixed_success_and_failure(store: SessionStore) -> None:
    """Successful and failed facts are tracked independently."""
    store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    extracted = [
        ExtractedFact(content="New fact.", scope="repo", action=FactAction.NEW),
        ExtractedFact(
            content="Uses pytest.",
            scope="repo",
            action=FactAction.CONFIRM,
            existing_content="Uses pytest.",  # exact match
        ),
        ExtractedFact(
            content="Uses ruff.",
            scope="repo",
            action=FactAction.SUPERSEDE,
            existing_content="Uses pytst.",  # typo — fails
        ),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert pr.added == 1
    assert pr.confirmed == 1
    assert pr.superseded == 0
    assert len(pr.failed) == 1


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — scope handling
# ═══════════════════════════════════════════════════════════════════════


def test_repo_scope_without_repo_skipped(store: SessionStore) -> None:
    """Repo-scoped facts are skipped when no repo is connected."""
    extracted = [
        ExtractedFact(content="Uses pytest.", scope="repo", action=FactAction.NEW),
        ExtractedFact(content="Prefers British English.", scope="global", action=FactAction.NEW),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, repo_scope=None)
    assert pr.added == 1  # Only the global fact.
    assert len(store.load_active_facts(GLOBAL)) == 1
    assert len(store.load_active_facts(RBTR_KEY)) == 0


def test_global_scope_always_works(store: SessionStore) -> None:
    """Global facts are inserted regardless of repo connection."""
    extracted = [
        ExtractedFact(content="Prefers terse comments.", scope="global", action=FactAction.NEW),
    ]
    pr = process_extracted_facts(extracted, store, SESSION_ID, repo_scope=None)
    assert pr.added == 1


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — empty input
# ═══════════════════════════════════════════════════════════════════════


def test_empty_extraction(store: SessionStore) -> None:
    """Empty fact list produces no changes."""
    pr = process_extracted_facts([], store, SESSION_ID, RBTR_KEY)
    assert (pr.added, pr.confirmed, pr.superseded) == (0, 0, 0)


# ═══════════════════════════════════════════════════════════════════════
# User prompt building
# ═══════════════════════════════════════════════════════════════════════


def test_prompt_includes_conversation() -> None:
    """Rendered prompt includes the conversation text."""
    prompt = _build_user_prompt("## User\nWhat testing framework?", [])
    assert "What testing framework?" in prompt


def test_prompt_includes_existing_facts(store: SessionStore) -> None:
    """Rendered prompt includes existing facts for LLM dedup context."""
    f1 = store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)
    f2 = store.insert_fact(GLOBAL, "Prefers British English.", SESSION_ID)
    facts = [f1, f2]

    prompt = _build_user_prompt("## User\nHello", facts)
    assert "- Uses pytest." in prompt
    assert "- Prefers British English." in prompt
    # No IDs exposed to the LLM.
    assert "id=" not in prompt


def test_prompt_empty_facts() -> None:
    """Rendered prompt shows '(none)' when no existing facts."""
    prompt = _build_user_prompt("## User\nHello", [])
    assert "(none)" in prompt


# ═══════════════════════════════════════════════════════════════════════
# FactExtractionResult model
# ═══════════════════════════════════════════════════════════════════════


def test_extraction_result_empty() -> None:
    """FactExtractionResult with no facts is valid."""
    result = FactExtractionResult()
    assert result.facts == []


def test_extraction_result_roundtrip() -> None:
    """FactExtractionResult serialises and deserialises cleanly."""
    result = FactExtractionResult(
        facts=[
            ExtractedFact(content="Uses pytest.", scope="repo", action=FactAction.NEW),
            ExtractedFact(
                content="Python 3.13+.",
                scope="repo",
                action=FactAction.SUPERSEDE,
                existing_content="Python 3.12.",
            ),
        ]
    )
    data = result.model_dump()
    restored = FactExtractionResult.model_validate(data)
    assert len(restored.facts) == 2
    assert restored.facts[1].existing_content == "Python 3.12."


# ═══════════════════════════════════════════════════════════════════════
# render_facts_instruction
# ═══════════════════════════════════════════════════════════════════════


def test_inject_global_and_repo(store: SessionStore) -> None:
    """Both global and repo facts appear in the rendered instruction."""
    store.insert_fact(GLOBAL, "Prefers British English.", SESSION_ID)
    store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    assert "Prefers British English." in result
    assert "Uses pytest." in result
    assert "## Facts" in result


def test_inject_global_only_no_repo(store: SessionStore) -> None:
    """Only global facts when no repo is connected."""
    store.insert_fact(GLOBAL, "Prefers British English.", SESSION_ID)
    store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    result = render_facts_instruction(store, repo_scope=None, max_facts=20, max_tokens=2000)
    assert "Prefers British English." in result
    assert "Uses pytest." not in result


def test_inject_most_recently_confirmed_first(store: SessionStore) -> None:
    """Facts are ordered by last_confirmed_at descending."""
    f1 = store.insert_fact(RBTR_KEY, "First fact.", SESSION_ID)
    store.insert_fact(RBTR_KEY, "Second fact.", SESSION_ID)
    # Confirm f1 so it becomes most recent.
    store.confirm_fact(f1.id)

    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    pos_first = result.index("First fact.")
    pos_second = result.index("Second fact.")
    assert pos_first < pos_second


def test_inject_fact_count_cap(store: SessionStore) -> None:
    """Respects max_facts limit."""
    for i in range(10):
        store.insert_fact(RBTR_KEY, f"Fact number {i}.", SESSION_ID)

    result = render_facts_instruction(store, RBTR_KEY, max_facts=3, max_tokens=2000)
    fact_lines = [line for line in result.splitlines() if line.startswith("- [")]
    assert len(fact_lines) == 3


def test_inject_token_cap(store: SessionStore) -> None:
    """Stops adding facts when token cap is exceeded."""
    for i in range(20):
        store.insert_fact(RBTR_KEY, f"This is fact number {i} with some extra words.", SESSION_ID)

    # Very small token cap — should only fit a few facts.
    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=50)
    fact_lines = [line for line in result.splitlines() if line.startswith("- [")]
    assert len(fact_lines) < 20
    assert len(fact_lines) >= 1


def test_inject_empty_returns_empty(store: SessionStore) -> None:
    """No facts returns empty string."""
    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    assert result == ""


def test_inject_scope_labels(store: SessionStore) -> None:
    """Fact lines include scope labels but not IDs."""
    store.insert_fact(GLOBAL, "Global pref.", SESSION_ID)
    store.insert_fact(RBTR_KEY, "Repo pref.", SESSION_ID)

    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    assert "[global] Global pref." in result
    assert f"[{RBTR_KEY}] Repo pref." in result
    assert "id=" not in result


def test_inject_excludes_superseded(store: SessionStore) -> None:
    """Superseded facts are not injected."""
    old = store.insert_fact(RBTR_KEY, "Python 3.12.", SESSION_ID)
    new = store.insert_fact(RBTR_KEY, "Python 3.13+.", SESSION_ID)
    store.supersede_fact(old.id, new.id)

    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    assert "Python 3.13+." in result
    assert "Python 3.12." not in result


# ═══════════════════════════════════════════════════════════════════════
# _build_clarify_prompt
# ═══════════════════════════════════════════════════════════════════════


def test_clarify_prompt_lists_failures() -> None:
    """Clarification prompt lists the failed facts."""
    failed = [
        ExtractedFact(
            content="Uses pytest with coverage.",
            scope="repo",
            action=FactAction.SUPERSEDE,
            existing_content="Uses pytst.",
        ),
    ]
    prompt = _build_clarify_prompt(failed)
    assert "Uses pytst." in prompt
    assert "Uses pytest with coverage." in prompt
    assert "didn't exactly match" in prompt


# ═══════════════════════════════════════════════════════════════════════
# _clarify_failed_facts
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.anyio
async def test_clarify_returns_corrected_facts(mocker: MockerFixture) -> None:
    """Clarification uses fact_extract_agent with message_history and returns corrections."""
    corrected = FactExtractionResult(
        facts=[
            ExtractedFact(
                content="Uses pytest with coverage.",
                scope="repo",
                action=FactAction.SUPERSEDE,
                existing_content="Uses pytest.",  # corrected
            ),
        ]
    )
    mock_run = mocker.patch(
        "rbtr.llm.memory.fact_extract_agent.run",
        return_value=_FakeAgentResult(corrected),
    )
    failed = [
        ExtractedFact(
            content="Uses pytest with coverage.",
            scope="repo",
            action=FactAction.SUPERSEDE,
            existing_content="Uses pytst.",  # typo
        ),
    ]
    history: list[object] = []  # Simulates result.all_messages() from first run.
    result = await _clarify_failed_facts(failed, history, mocker.MagicMock(), None)
    assert len(result.facts) == 1
    assert result.facts[0].existing_content == "Uses pytest."
    mock_run.assert_called_once()
    # Verify message_history was passed.
    _, kwargs = mock_run.call_args
    assert kwargs["message_history"] is history


@pytest.mark.anyio
async def test_clarify_model_failure_returns_empty(mocker: MockerFixture) -> None:
    """Model error during clarification returns empty result."""
    from pydantic_ai.exceptions import ModelHTTPError

    mocker.patch(
        "rbtr.llm.memory.fact_extract_agent.run",
        side_effect=ModelHTTPError(status_code=500, model_name="test", body="fail"),
    )
    result = await _clarify_failed_facts([], [], mocker.MagicMock(), None)
    assert result.facts == []
    assert result.input_tokens == 0


@pytest.mark.anyio
async def test_clarify_empty_response(mocker: MockerFixture) -> None:
    """Model returns no corrected facts."""
    mocker.patch(
        "rbtr.llm.memory.fact_extract_agent.run",
        return_value=_FakeAgentResult(FactExtractionResult()),
    )
    result = await _clarify_failed_facts(
        [ExtractedFact(content="X.", action=FactAction.SUPERSEDE, existing_content="Y.")],
        [],
        mocker.MagicMock(),
        None,
    )
    assert result.facts == []


# ── Helpers ──────────────────────────────────────────────────────────


class _FakeAgentResult:
    """Minimal stand-in for ``AgentRunResult``."""

    def __init__(self, output: FactExtractionResult) -> None:
        self.output = output

    def usage(self) -> _FakeUsage:
        return _FakeUsage()

    def new_messages(self) -> list[object]:
        return []


class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
