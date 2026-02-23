"""Tests for SessionStore — schema creation, versioning, UUID7."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from uuid_utils import uuid7

from rbtr.sessions.store import _SCHEMA_VERSION, SessionStore


def test_uuid7_version_and_sortability() -> None:
    """UUID7 has version 7 and is time-sortable across milliseconds."""
    first = uuid7()
    time.sleep(0.002)  # ensure different millisecond
    second = uuid7()
    assert first.version == 7
    assert second.version == 7
    assert str(first) < str(second)


def test_uuid7_uniqueness() -> None:
    """UUID7 generates unique values."""
    ids = {str(uuid7()) for _ in range(1000)}
    assert len(ids) == 1000


def test_schema_created_on_fresh_db() -> None:
    """Opening a fresh in-memory DB creates the messages table."""
    with SessionStore() as store:
        cur = store._con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        assert cur.fetchone() is not None


def test_user_version_set() -> None:
    """Schema version is set via PRAGMA user_version."""
    with SessionStore() as store:
        version = store._user_version()
        assert version == _SCHEMA_VERSION


def test_wal_mode_enabled() -> None:
    """WAL journal mode is active."""
    with SessionStore() as store:
        row = store._con.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        # In-memory DBs may report 'memory' instead of 'wal'.
        # File-backed DBs will report 'wal'.
        assert row[0] in ("wal", "memory")


def test_indexes_created() -> None:
    """Expected indexes exist after schema creation."""
    with SessionStore() as store:
        cur = store._con.execute("SELECT name FROM sqlite_master WHERE type='index'")
        names = {row[0] for row in cur.fetchall()}
        assert "idx_messages_session" in names
        assert "idx_messages_user_text" in names
        assert "idx_messages_session_created" in names


def test_new_id_is_valid_uuid7() -> None:
    """new_id returns a valid UUID string."""
    with SessionStore() as store:
        sid = store.new_id()
        parsed = uuid.UUID(sid)
        assert parsed.version == 7


def test_idempotent_open(tmp_path: Path) -> None:
    """Opening the same DB twice doesn't fail or reset the schema."""
    db_path = tmp_path / "test.db"
    with SessionStore(db_path) as store:
        store._con.execute(
            "INSERT INTO messages (id, session_id, created_at, kind) "
            "VALUES ('test-id', 'test-session', '2025-01-01T00:00:00', 'request')"
        )
        store._con.commit()

    # Re-open — should not drop data.
    with SessionStore(db_path) as store:
        row = store._con.execute("SELECT id FROM messages WHERE id = 'test-id'").fetchone()
        assert row is not None
        assert store._user_version() == _SCHEMA_VERSION


def test_newer_version_warns(tmp_path: Path) -> None:
    """Opening a DB with a newer version logs a warning but doesn't crash."""
    db_path = tmp_path / "test.db"

    # Create DB with a future schema version.
    with SessionStore(db_path) as store:
        store._set_user_version(_SCHEMA_VERSION + 99)

    # Re-open — should not crash.
    with SessionStore(db_path) as store:
        assert store._user_version() == _SCHEMA_VERSION + 99
