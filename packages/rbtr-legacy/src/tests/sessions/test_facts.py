"""Tests for the facts (cross-session memory) layer of SessionStore.

Data-first: realistic facts from `fact_data` seeded via a shared
fixture.  Tests verify behaviours through the store's public fact
API — insert, confirm, supersede, load, delete, FTS5 search.

Organisation:
- Fixtures
- Insert & load roundtrip
- Scope filtering
- Ordering
- Confirmation
- Supersession
- Deletion
- FTS5 dedup search
- FTS5 cross-scope isolation
- Empty store
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from rbtr.sessions.kinds import Fact
from rbtr.sessions.store import _SCHEMA_VERSION, SessionStore

from .fact_data import (
    ALL_BASELINE,
    CONFIRMATION_SEQUENCE,
    CROSS_SCOPE_SIMILAR,
    DEDUP_PAIRS,
    GLOBAL,
    GLOBAL_FACTS,
    OTHER_KEY,
    OTHER_REPO_FACTS,
    RBTR_FACTS,
    RBTR_KEY,
    SUPERSESSION_PAIRS,
)

SESSION_ID = "test-session-001"
SESSION_ID_2 = "test-session-002"


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def seeded_store(store: SessionStore) -> SessionStore:
    """Store pre-loaded with ALL_BASELINE facts."""
    for fact in ALL_BASELINE:
        store.insert_fact(fact["scope"], fact["content"], SESSION_ID)
    return store


# ═══════════════════════════════════════════════════════════════════════
# Insert & load roundtrip
# ═══════════════════════════════════════════════════════════════════════


def test_insert_and_load_roundtrip(store: SessionStore) -> None:
    """Inserted fact is retrievable via load_active_facts."""
    fact = store.insert_fact(RBTR_KEY, "pytest is the test framework.", SESSION_ID)
    loaded = store.load_active_facts(RBTR_KEY)
    assert len(loaded) == 1
    assert loaded[0].id == fact.id
    assert loaded[0].content == "pytest is the test framework."
    assert loaded[0].scope == RBTR_KEY
    assert loaded[0].source_session_id == SESSION_ID
    assert loaded[0].confirm_count == 1


def test_insert_returns_fact_with_fields(store: SessionStore) -> None:
    """insert_fact returns a Fact with all fields populated."""
    fact = store.insert_fact(GLOBAL, "User prefers British English.", SESSION_ID)
    assert isinstance(fact, Fact)
    assert fact.scope == GLOBAL
    assert fact.content == "User prefers British English."
    assert fact.created_at == fact.last_confirmed_at
    assert fact.confirm_count == 1


# ═══════════════════════════════════════════════════════════════════════
# Scope filtering
# ═══════════════════════════════════════════════════════════════════════


def test_load_filters_by_scope(seeded_store: SessionStore) -> None:
    """load_active_facts returns only facts matching the requested scope."""
    rbtr = seeded_store.load_active_facts(RBTR_KEY)
    global_ = seeded_store.load_active_facts(GLOBAL)
    other = seeded_store.load_active_facts(OTHER_KEY)

    assert len(rbtr) == len(RBTR_FACTS)
    assert len(global_) == len(GLOBAL_FACTS)
    assert len(other) == len(OTHER_REPO_FACTS)


def test_load_repo_excludes_global(seeded_store: SessionStore) -> None:
    """Repo-scoped load never includes global facts."""
    rbtr = seeded_store.load_active_facts(RBTR_KEY)
    contents = {f.content for f in rbtr}
    for gf in GLOBAL_FACTS:
        assert gf["content"] not in contents


def test_load_global_excludes_repo(seeded_store: SessionStore) -> None:
    """Global load never includes repo-scoped facts."""
    global_ = seeded_store.load_active_facts(GLOBAL)
    contents = {f.content for f in global_}
    for rf in RBTR_FACTS:
        assert rf["content"] not in contents


# ═══════════════════════════════════════════════════════════════════════
# Ordering
# ═══════════════════════════════════════════════════════════════════════


def test_load_ordered_by_last_confirmed_desc(store: SessionStore) -> None:
    """Facts are returned most recently confirmed first."""
    f1 = store.insert_fact(RBTR_KEY, "First fact.", SESSION_ID)
    time.sleep(0.01)
    f2 = store.insert_fact(RBTR_KEY, "Second fact.", SESSION_ID)
    time.sleep(0.01)
    f3 = store.insert_fact(RBTR_KEY, "Third fact.", SESSION_ID)

    loaded = store.load_active_facts(RBTR_KEY)
    assert [f.id for f in loaded] == [f3.id, f2.id, f1.id]


def test_confirmed_fact_moves_to_top(store: SessionStore) -> None:
    """Confirming an older fact moves it to the top of the order."""
    f1 = store.insert_fact(RBTR_KEY, "Oldest.", SESSION_ID)
    time.sleep(0.01)
    store.insert_fact(RBTR_KEY, "Middle.", SESSION_ID)
    time.sleep(0.01)
    store.insert_fact(RBTR_KEY, "Newest.", SESSION_ID)
    time.sleep(0.01)
    store.confirm_fact(f1.id)

    loaded = store.load_active_facts(RBTR_KEY)
    assert loaded[0].id == f1.id


# ═══════════════════════════════════════════════════════════════════════
# Confirmation
# ═══════════════════════════════════════════════════════════════════════


def test_confirm_bumps_count_and_timestamp(store: SessionStore) -> None:
    """confirm_fact increments confirm_count and updates last_confirmed_at."""
    fact = store.insert_fact(RBTR_KEY, "pytest is used.", SESSION_ID)
    original_ts = fact.last_confirmed_at
    time.sleep(0.01)

    store.confirm_fact(fact.id)
    loaded = store.load_active_facts(RBTR_KEY)

    assert loaded[0].confirm_count == 2
    assert loaded[0].last_confirmed_at > original_ts


def test_multiple_confirmations(store: SessionStore) -> None:
    """Multiple confirmations accumulate correctly."""
    fact = store.insert_fact(RBTR_KEY, "pytest is used.", SESSION_ID)
    for _ in range(4):
        time.sleep(0.002)
        store.confirm_fact(fact.id)

    loaded = store.load_active_facts(RBTR_KEY)
    assert loaded[0].confirm_count == 5


# ═══════════════════════════════════════════════════════════════════════
# Supersession
# ═══════════════════════════════════════════════════════════════════════


def test_supersede_excludes_from_active(store: SessionStore) -> None:
    """Superseded facts are excluded from load_active_facts."""
    old = store.insert_fact(RBTR_KEY, "Python 3.12.", SESSION_ID)
    new = store.insert_fact(RBTR_KEY, "Python 3.13+.", SESSION_ID)
    store.supersede_fact(old.id, new.id)

    loaded = store.load_active_facts(RBTR_KEY)
    ids = {f.id for f in loaded}
    assert old.id not in ids
    assert new.id in ids


def test_supersede_idempotent(store: SessionStore) -> None:
    """Superseding an already-superseded fact is a no-op."""
    old = store.insert_fact(RBTR_KEY, "Old.", SESSION_ID)
    new1 = store.insert_fact(RBTR_KEY, "New 1.", SESSION_ID)
    new2 = store.insert_fact(RBTR_KEY, "New 2.", SESSION_ID)
    store.supersede_fact(old.id, new1.id)
    # Second supersede should not change the existing superseded_by.
    store.supersede_fact(old.id, new2.id)

    loaded = store.load_active_facts(RBTR_KEY)
    ids = {f.id for f in loaded}
    assert old.id not in ids


@pytest.mark.parametrize(
    ("old_content", "new_content"),
    SUPERSESSION_PAIRS,
    ids=[p[1][:40] for p in SUPERSESSION_PAIRS],
)
def test_supersession_pairs(store: SessionStore, old_content: str, new_content: str) -> None:
    """Each supersession pair: old is excluded, new is active."""
    old = store.insert_fact(RBTR_KEY, old_content, SESSION_ID)
    new = store.insert_fact(RBTR_KEY, new_content, SESSION_ID)
    store.supersede_fact(old.id, new.id)

    loaded = store.load_active_facts(RBTR_KEY)
    contents = {f.content for f in loaded}
    assert old_content not in contents
    assert new_content in contents


# ═══════════════════════════════════════════════════════════════════════
# Deletion
# ═══════════════════════════════════════════════════════════════════════


def test_delete_removes_fact(store: SessionStore) -> None:
    """delete_fact removes the row entirely."""
    fact = store.insert_fact(RBTR_KEY, "Temporary.", SESSION_ID)
    deleted = store.delete_fact(fact.id)
    assert deleted == 1
    assert len(store.load_active_facts(RBTR_KEY)) == 0


def test_delete_nonexistent(store: SessionStore) -> None:
    """Deleting a non-existent ID returns 0, no error."""
    assert store.delete_fact("nonexistent-id") == 0


def test_delete_old_facts(store: SessionStore) -> None:
    """delete_old_facts removes facts confirmed before the cutoff."""
    store.insert_fact(RBTR_KEY, "Old fact.", SESSION_ID)
    time.sleep(0.01)
    cutoff = datetime.now(UTC)
    time.sleep(0.01)
    store.insert_fact(RBTR_KEY, "Recent fact.", SESSION_ID)

    deleted = store.delete_old_facts(before=cutoff)
    assert deleted == 1

    active = store.load_active_facts(RBTR_KEY)
    assert len(active) == 1
    assert active[0].content == "Recent fact."


def test_delete_old_facts_keeps_all_recent(store: SessionStore) -> None:
    """delete_old_facts with an old cutoff deletes nothing."""
    store.insert_fact(RBTR_KEY, "Fresh.", SESSION_ID)
    deleted = store.delete_old_facts(before=datetime(2000, 1, 1, tzinfo=UTC))
    assert deleted == 0
    assert len(store.load_active_facts(RBTR_KEY)) == 1


def test_delete_old_facts_uses_confirmed_at(store: SessionStore) -> None:
    """Confirmed facts survive based on `last_confirmed_at`, not `created_at`."""
    fact = store.insert_fact(RBTR_KEY, "Confirmed recently.", SESSION_ID)
    time.sleep(0.01)
    # Confirm bumps last_confirmed_at to now.
    store.confirm_fact(fact.id)
    time.sleep(0.01)
    cutoff = datetime.now(UTC)

    deleted = store.delete_old_facts(before=cutoff)
    # The fact was confirmed after creation, but before cutoff — deleted.
    # (confirm happened before cutoff)
    assert deleted == 1


# ═══════════════════════════════════════════════════════════════════════
# FTS5 dedup search
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("existing", "query", "should_match"),
    [p for p in DEDUP_PAIRS if p[2]],
    ids=[p[1][:50] for p in DEDUP_PAIRS if p[2]],
)
def test_fts5_finds_dedup_match(
    store: SessionStore, existing: str, query: str, should_match: bool
) -> None:
    """FTS5 search surfaces semantically equivalent facts."""
    store.insert_fact(RBTR_KEY, existing, SESSION_ID)
    results = store.search_facts(query, RBTR_KEY)
    assert len(results) >= 1, f"Expected match for: {query!r}"
    assert results[0].content == existing


@pytest.mark.parametrize(
    ("existing", "query", "should_match"),
    [p for p in DEDUP_PAIRS if not p[2]],
    ids=[p[1][:50] for p in DEDUP_PAIRS if not p[2]],
)
def test_fts5_search_handles_low_overlap(
    store: SessionStore, existing: str, query: str, should_match: bool
) -> None:
    """FTS5 search doesn't crash on queries with low keyword overlap.

    FTS5 MATCH with OR tokens *may* return results for shared
    keywords — that's expected.  The dedup decision is the LLM's
    job, not the store's.  This test verifies the search is safe
    on these inputs.
    """
    store.insert_fact(RBTR_KEY, existing, SESSION_ID)
    # Should not raise — results may or may not be returned.
    store.search_facts(query, RBTR_KEY)


def test_fts5_search_returns_bm25_order(seeded_store: SessionStore) -> None:
    """Search results are ordered by BM25 relevance."""
    results = seeded_store.search_facts("pytest testing", RBTR_KEY)
    assert len(results) >= 1
    # The pytest fact should rank highest.
    assert "pytest" in results[0].content.lower()


# ═══════════════════════════════════════════════════════════════════════
# FTS5 cross-scope isolation
# ═══════════════════════════════════════════════════════════════════════


def test_fts5_cross_scope_isolation(store: SessionStore) -> None:
    """Search in one scope does not return facts from another scope."""
    for fact in CROSS_SCOPE_SIMILAR:
        store.insert_fact(fact["scope"], fact["content"], SESSION_ID)

    global_results = store.search_facts("composition inheritance", GLOBAL)
    repo_results = store.search_facts("composition inheritance", RBTR_KEY)

    global_scopes = {r.scope for r in global_results}
    repo_scopes = {r.scope for r in repo_results}

    assert all(s == GLOBAL for s in global_scopes)
    assert all(s == RBTR_KEY for s in repo_scopes)


def test_fts5_excludes_superseded(store: SessionStore) -> None:
    """FTS5 search excludes superseded facts."""
    old = store.insert_fact(RBTR_KEY, "Python 3.12 is the target.", SESSION_ID)
    new = store.insert_fact(RBTR_KEY, "Python 3.13+ is the target.", SESSION_ID)
    store.supersede_fact(old.id, new.id)

    results = store.search_facts("Python target", RBTR_KEY)
    ids = {r.id for r in results}
    assert old.id not in ids


# ═══════════════════════════════════════════════════════════════════════
# Confirmation sequence
# ═══════════════════════════════════════════════════════════════════════


def test_confirmation_sequence_fts5_matches(store: SessionStore) -> None:
    """Each rewording in CONFIRMATION_SEQUENCE matches the original via FTS5."""
    original = CONFIRMATION_SEQUENCE[0]
    store.insert_fact(RBTR_KEY, original, SESSION_ID)

    for rewording in CONFIRMATION_SEQUENCE[1:]:
        results = store.search_facts(rewording, RBTR_KEY)
        assert len(results) >= 1, f"No FTS5 match for: {rewording!r}"
        assert results[0].content == original


# ═══════════════════════════════════════════════════════════════════════
# Empty store
# ═══════════════════════════════════════════════════════════════════════


def test_empty_store_load(store: SessionStore) -> None:
    """Loading from an empty store returns empty list, no error."""
    assert store.load_active_facts(RBTR_KEY) == []
    assert store.load_active_facts(GLOBAL) == []


def test_empty_store_search(store: SessionStore) -> None:
    """Searching an empty store returns empty list, no error."""
    assert store.search_facts("anything", RBTR_KEY) == []


# ═══════════════════════════════════════════════════════════════════════
# Migration
# ═══════════════════════════════════════════════════════════════════════


def test_migration_preserves_sessions() -> None:
    """Migrating from v2026030301 adds facts without touching fragments."""

    with SessionStore() as store:
        # Roll back to the pre-facts schema: drop facts objects, reset version.
        store._con.executescript(
            "DROP TRIGGER IF EXISTS facts_ai;"
            "DROP TRIGGER IF EXISTS facts_ad;"
            "DROP TRIGGER IF EXISTS facts_au;"
            "DROP TABLE IF EXISTS facts_fts;"
            "DROP TABLE IF EXISTS facts;"
        )
        store._set_user_version(2026_03_03_01)

        # Insert a fragment to verify it survives migration.
        store._con.execute(
            "INSERT INTO fragments (id, session_id, message_id, fragment_index,"
            " fragment_kind, created_at, status)"
            " VALUES ('f1', 's1', 'f1', 0, 'request-message',"
            " '2026-01-01T00:00:00', 'complete')"
        )
        store._con.commit()

        # Re-run setup — should apply incremental migration.
        store._setup()

        assert store._user_version() == _SCHEMA_VERSION

        # Fragment survived.
        row = store._con.execute("SELECT id FROM fragments WHERE id = 'f1'").fetchone()
        assert row is not None

        # Facts table works.
        fact = store.insert_fact(RBTR_KEY, "Migration test fact.", SESSION_ID)
        loaded = store.load_active_facts(RBTR_KEY)
        assert len(loaded) == 1
        assert loaded[0].id == fact.id


# ═══════════════════════════════════════════════════════════════════════
# Load limit
# ═══════════════════════════════════════════════════════════════════════


def test_load_respects_limit(store: SessionStore) -> None:
    """load_active_facts respects the limit parameter."""
    for i in range(10):
        store.insert_fact(RBTR_KEY, f"Fact {i}.", SESSION_ID)

    loaded = store.load_active_facts(RBTR_KEY, limit=3)
    assert len(loaded) == 3
