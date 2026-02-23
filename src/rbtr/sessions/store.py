"""SQLite-backed session store for conversation persistence.

Single ``messages`` table: sessions are a grouping column
(``session_id``), not a separate entity.  Session-level metadata
is denormalised onto every row.  Aggregates are computed via
``GROUP BY``.

Schema versioning uses ``PRAGMA user_version``.  Migrations are
hand-written functions run sequentially on open.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage
from uuid_utils import uuid7

from rbtr.constants import RBTR_DIR
from rbtr.sessions.serialise import MessageRow, deserialise_message

log = logging.getLogger(__name__)

SESSIONS_DB_PATH = RBTR_DIR / "sessions.db"

_SCHEMA_VERSION = 1

# ── SQL loader ───────────────────────────────────────────────────────

_sql_pkg = importlib.resources.files("rbtr.sessions") / "sql"


def _load_sql(name: str) -> str:
    """Read a ``.sql`` file from the ``sql/`` package directory."""
    return (_sql_pkg / name).read_text(encoding="utf-8")


_SCHEMA_SQL = _load_sql("schema.sql")
_INSERT_MESSAGE_SQL = _load_sql("insert_message.sql")
_LOAD_MESSAGES_SQL = _load_sql("load_messages.sql")
_LIST_SESSIONS_SQL = _load_sql("list_sessions.sql")
_DELETE_SESSION_SQL = _load_sql("delete_session.sql")
_DELETE_OLD_SESSIONS_SQL = _load_sql("delete_old_sessions.sql")
_SEARCH_HISTORY_SQL = _load_sql("search_history.sql")


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Lightweight session listing — no message bodies."""

    session_id: str
    session_label: str | None
    last_active: str
    message_count: int
    total_cost: float


# ── Store ────────────────────────────────────────────────────────────


class SessionStore:
    """SQLite-backed conversation store.

    Thread-safe: a ``threading.Lock`` serialises all writes.  Reads
    use WAL mode for non-blocking concurrency.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  ``None`` creates an
        in-memory database (useful for tests).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is not None:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        uri = str(db_path) if db_path is not None else ":memory:"
        self._con = sqlite3.connect(uri, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._setup()

    # ── Setup & migrations ───────────────────────────────────────────

    def _setup(self) -> None:
        """Apply schema and run migrations if needed."""
        self._con.execute("PRAGMA journal_mode = WAL")
        self._con.execute("PRAGMA synchronous = NORMAL")
        self._con.execute("PRAGMA foreign_keys = ON")

        version = self._user_version()
        if version == 0:
            # Fresh database — apply schema and set version.
            self._con.executescript(_SCHEMA_SQL)
            self._set_user_version(_SCHEMA_VERSION)
            log.debug("sessions: created schema v%d", _SCHEMA_VERSION)
        elif version < _SCHEMA_VERSION:
            self._migrate(version)
        elif version > _SCHEMA_VERSION:
            log.warning(
                "sessions: DB version %d is newer than code version %d — "
                "proceeding, but data may be incompatible",
                version,
                _SCHEMA_VERSION,
            )

    def _user_version(self) -> int:
        row = self._con.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def _set_user_version(self, version: int) -> None:
        # PRAGMA doesn't support parameter binding.
        self._con.execute(f"PRAGMA user_version = {version}")

    def _migrate(self, from_version: int) -> None:
        """Run sequential migrations from *from_version* to current.

        Each migration is a function ``_migrate_vN_to_vM(con)`` that
        applies DDL changes inside a transaction.
        """
        current = from_version
        while current < _SCHEMA_VERSION:
            target = current + 1
            fn_name = f"_migrate_v{current}_to_v{target}"
            fn = globals().get(fn_name)
            if fn is None:
                msg = f"Missing migration function: {fn_name}"
                raise RuntimeError(msg)
            log.info("sessions: migrating v%d → v%d", current, target)
            fn(self._con)
            self._set_user_version(target)
            current = target

    # ── Public API ───────────────────────────────────────────────────

    def new_id(self) -> str:
        """Generate a new UUID7 identifier (for sessions or rows)."""
        return str(uuid7())

    # ── Writes ───────────────────────────────────────────────────────

    def save_messages(self, rows: list[MessageRow]) -> None:
        """Bulk insert message rows (append-only).

        Uses a single transaction for atomicity.
        """
        if not rows:
            return
        with self._lock, self._con:
            self._con.executemany(
                _INSERT_MESSAGE_SQL,
                [dataclasses.astuple(r) for r in rows],
            )

    def mark_compacted(self, message_ids: list[str], summary_id: str) -> None:
        """Set ``compacted_by`` on the given message rows.

        Called after compaction to link old messages to their summary.
        """
        if not message_ids:
            return
        placeholders = ",".join("?" * len(message_ids))
        sql = f"UPDATE messages SET compacted_by = ? WHERE id IN ({placeholders})"  # noqa: S608  # placeholders are only '?' chars
        with self._lock, self._con:
            self._con.execute(sql, [summary_id, *message_ids])

    def delete_session(self, session_id: str) -> int:
        """Delete all messages for a session.  Returns rows deleted."""
        with self._lock, self._con:
            cur = self._con.execute(_DELETE_SESSION_SQL, [session_id])
            return cur.rowcount

    def delete_old_sessions(self, before: datetime) -> int:
        """Delete sessions whose last activity is before *before*.

        Returns the number of rows deleted.
        """
        with self._lock, self._con:
            cur = self._con.execute(
                _DELETE_OLD_SESSIONS_SQL,
                [before.isoformat()],
            )
            return cur.rowcount

    # ── Reads ────────────────────────────────────────────────────────

    def load_messages(self, session_id: str) -> list[ModelMessage]:
        """Load active (non-compacted) messages for a session.

        Returns deserialised ``ModelMessage`` objects in chronological
        order.  Corrupt rows are logged and skipped.
        """
        rows = self._con.execute(_LOAD_MESSAGES_SQL, [session_id]).fetchall()
        messages: list[ModelMessage] = []
        for row in rows:
            try:
                messages.append(deserialise_message(row["message_json"]))
            except (ValidationError, ValueError, KeyError):
                log.warning("sessions: skipping corrupt message in %s", session_id)
        return messages

    def list_sessions(
        self,
        *,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        """List sessions, most recent first.

        Optionally filter by repo.  Returns lightweight summaries.
        """
        rows = self._con.execute(
            _LIST_SESSIONS_SQL,
            [repo_owner, repo_owner, repo_name, repo_name, limit],
        ).fetchall()
        return [
            SessionSummary(
                session_id=row["session_id"],
                session_label=row["session_label"],
                last_active=row["last_active"],
                message_count=row["message_count"],
                total_cost=float(row["total_cost"]),
            )
            for row in rows
        ]

    def search_history(
        self,
        prefix: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """Search user input history across all sessions.

        Returns deduplicated ``user_text`` values, most recent first.
        """
        rows = self._con.execute(
            _SEARCH_HISTORY_SQL,
            [prefix, prefix, limit],
        ).fetchall()
        return [row["user_text"] for row in rows]

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
