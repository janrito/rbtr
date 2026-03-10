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
- ExtractionResult model
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from rbtr.llm.memory import (
    ExtractedFact,
    ExtractionResult,
    FactAction,
    _build_user_prompt,
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
    added, confirmed, superseded = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert added == 3
    assert confirmed == 0
    assert superseded == 0

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
    added, confirmed, _ = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert added == 1
    assert confirmed == 0

    # Two facts now — the LLM should have tagged it 'confirm',
    # but if it didn't, pruning handles the duplicate later.
    facts = store.load_active_facts(RBTR_KEY)
    assert len(facts) == 2


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — confirm
# ═══════════════════════════════════════════════════════════════════════


def test_confirm_bumps_existing(store: SessionStore) -> None:
    """Confirm action bumps an existing fact."""
    existing = store.insert_fact(RBTR_KEY, "Uses pytest.", SESSION_ID)

    extracted = [
        ExtractedFact(
            content="Uses pytest.",
            scope="repo",
            action=FactAction.CONFIRM,
            existing_id=existing.id,
        ),
    ]
    added, confirmed, _ = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert added == 0
    assert confirmed == 1

    facts = store.load_active_facts(RBTR_KEY)
    assert len(facts) == 1
    assert facts[0].confirm_count == 2


def test_confirm_without_existing_id_falls_through(store: SessionStore) -> None:
    """Confirm without existing_id is treated as new (fallback)."""
    extracted = [
        ExtractedFact(content="Orphan confirm.", scope="repo", action=FactAction.CONFIRM),
    ]
    added, confirmed, _ = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert added == 1
    assert confirmed == 0


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — supersede
# ═══════════════════════════════════════════════════════════════════════


def test_supersede_replaces_old(store: SessionStore) -> None:
    """Supersede action inserts new and marks old as superseded."""
    old = store.insert_fact(RBTR_KEY, "Python 3.12.", SESSION_ID)

    extracted = [
        ExtractedFact(
            content="Python 3.13+.",
            scope="repo",
            action=FactAction.SUPERSEDE,
            existing_id=old.id,
        ),
    ]
    added, _confirmed, superseded = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert added == 1
    assert superseded == 1

    facts = store.load_active_facts(RBTR_KEY)
    assert len(facts) == 1
    assert facts[0].content == "Python 3.13+."


def test_supersede_without_existing_id_falls_through(store: SessionStore) -> None:
    """Supersede without existing_id is treated as new (fallback)."""
    extracted = [
        ExtractedFact(content="Orphan supersede.", scope="repo", action=FactAction.SUPERSEDE),
    ]
    added, _, superseded = process_extracted_facts(extracted, store, SESSION_ID, RBTR_KEY)
    assert added == 1
    assert superseded == 0


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — scope handling
# ═══════════════════════════════════════════════════════════════════════


def test_repo_scope_without_repo_skipped(store: SessionStore) -> None:
    """Repo-scoped facts are skipped when no repo is connected."""
    extracted = [
        ExtractedFact(content="Uses pytest.", scope="repo", action=FactAction.NEW),
        ExtractedFact(content="Prefers British English.", scope="global", action=FactAction.NEW),
    ]
    added, _, _ = process_extracted_facts(extracted, store, SESSION_ID, repo_scope=None)
    assert added == 1  # Only the global fact.
    assert len(store.load_active_facts(GLOBAL)) == 1
    assert len(store.load_active_facts(RBTR_KEY)) == 0


def test_global_scope_always_works(store: SessionStore) -> None:
    """Global facts are inserted regardless of repo connection."""
    extracted = [
        ExtractedFact(content="Prefers terse comments.", scope="global", action=FactAction.NEW),
    ]
    added, _, _ = process_extracted_facts(extracted, store, SESSION_ID, repo_scope=None)
    assert added == 1


# ═══════════════════════════════════════════════════════════════════════
# process_extracted_facts — empty input
# ═══════════════════════════════════════════════════════════════════════


def test_empty_extraction(store: SessionStore) -> None:
    """Empty fact list produces no changes."""
    added, confirmed, superseded = process_extracted_facts([], store, SESSION_ID, RBTR_KEY)
    assert (added, confirmed, superseded) == (0, 0, 0)


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
    assert f"id={f1.id}" in prompt
    assert "Uses pytest." in prompt
    assert f"id={f2.id}" in prompt
    assert "Prefers British English." in prompt


def test_prompt_empty_facts() -> None:
    """Rendered prompt shows '(none)' when no existing facts."""
    prompt = _build_user_prompt("## User\nHello", [])
    assert "(none)" in prompt


# ═══════════════════════════════════════════════════════════════════════
# ExtractionResult model
# ═══════════════════════════════════════════════════════════════════════


def test_extraction_result_empty() -> None:
    """ExtractionResult with no facts is valid."""
    result = ExtractionResult()
    assert result.facts == []


def test_extraction_result_roundtrip() -> None:
    """ExtractionResult serialises and deserialises cleanly."""
    result = ExtractionResult(
        facts=[
            ExtractedFact(content="Uses pytest.", scope="repo", action=FactAction.NEW),
            ExtractedFact(
                content="Python 3.13+.",
                scope="repo",
                action=FactAction.SUPERSEDE,
                existing_id="old-fact-id",
            ),
        ]
    )
    data = result.model_dump()
    restored = ExtractionResult.model_validate(data)
    assert len(restored.facts) == 2
    assert restored.facts[1].existing_id == "old-fact-id"


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
    assert "## Learned facts" in result


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
    assert result.count("- [id=") == 3


def test_inject_token_cap(store: SessionStore) -> None:
    """Stops adding facts when token cap is exceeded."""
    for i in range(20):
        store.insert_fact(RBTR_KEY, f"This is fact number {i} with some extra words.", SESSION_ID)

    # Very small token cap — should only fit a few facts.
    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=50)
    fact_lines = [line for line in result.splitlines() if line.startswith("- [id=")]
    assert len(fact_lines) < 20
    assert len(fact_lines) >= 1


def test_inject_empty_returns_empty(store: SessionStore) -> None:
    """No facts returns empty string."""
    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    assert result == ""


def test_inject_scope_labels_and_ids(store: SessionStore) -> None:
    """Fact lines include ids and scope labels."""
    g = store.insert_fact(GLOBAL, "Global pref.", SESSION_ID)
    r = store.insert_fact(RBTR_KEY, "Repo pref.", SESSION_ID)

    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    assert f"[id={g.id}, global]" in result
    assert f"[id={r.id}, {RBTR_KEY}]" in result


def test_inject_excludes_superseded(store: SessionStore) -> None:
    """Superseded facts are not injected."""
    old = store.insert_fact(RBTR_KEY, "Python 3.12.", SESSION_ID)
    new = store.insert_fact(RBTR_KEY, "Python 3.13+.", SESSION_ID)
    store.supersede_fact(old.id, new.id)

    result = render_facts_instruction(store, RBTR_KEY, max_facts=20, max_tokens=2000)
    assert "Python 3.13+." in result
    assert "Python 3.12." not in result
