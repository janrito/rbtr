"""SQLite-backed session store for conversation persistence.

Single ``fragments`` table: sessions are a grouping column
(``session_id``), not a separate entity.  Each row is either a
message-level fragment (``message_id = id``, self-referencing) or
a content fragment (``message_id`` points to the message row).
Aggregates are computed via ``GROUP BY``.

Schema versioning uses ``PRAGMA user_version``.  Migrations are
hand-written functions run sequentially on open.

Public API
----------
- **read**: ``load_messages``, ``load_message_ids``, ``list_sessions``,
  ``search_history``
- **write**: ``save_messages``, ``save_incident``, ``begin_response``,
  ``compact_session``
- **delete**: ``delete_session``, ``delete_old_sessions``,
  ``delete_excess_sessions``
- **lifecycle**: ``new_id``, ``close``

Message write methods accept ``ModelMessage`` objects — serialisation
is fully internal.  Incident methods accept typed ``Incident``
payloads from ``serialise.py``.
"""

from __future__ import annotations

import dataclasses
import importlib.resources
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponsePart
from uuid_utils import uuid7

from rbtr.constants import RBTR_DIR
from rbtr.sessions.serialise import (
    Fragment,
    FragmentKind,
    FragmentStatus,
    Incident,
    SessionContext,
    dump_part,
    prepare_incident_row,
    prepare_input_row,
    prepare_message_row,
    prepare_part_row,
    prepare_part_rows,
    reconstruct_message,
)
from rbtr.sessions.stats import (
    GlobalStats,
    IncidentStats,
    TokenStats,
    ToolStat,
    global_incident_stats as _global_incident_stats,
    global_stats as _global_stats,
    incident_stats as _incident_stats,
    token_stats as _token_stats,
    tool_stats as _tool_stats,
)

log = logging.getLogger(__name__)

SESSIONS_DB_PATH = RBTR_DIR / "sessions.db"

# Date-based schema version: YYYYMMDD0R where R is a release
# counter for multiple migrations on the same day.  Fits in
# SQLite's 32-bit PRAGMA user_version (max 2_147_483_647).
_SCHEMA_VERSION = 2026_03_03_01

# ── SQL loader ───────────────────────────────────────────────────────

_sql_pkg = importlib.resources.files("rbtr.sessions") / "sql"


def _load_sql(name: str) -> str:
    """Read a ``.sql`` file from the ``sql/`` package directory."""
    return (_sql_pkg / name).read_text(encoding="utf-8")


_SCHEMA_SQL = _load_sql("schema.sql")
_INSERT_FRAGMENT_SQL = _load_sql("insert_fragment.sql")
_UPDATE_FRAGMENT_SQL = _load_sql("update_fragment.sql")
_LOAD_MESSAGES_SQL = _load_sql("load_messages.sql")
_LOAD_MESSAGE_IDS_SQL = _load_sql("load_message_ids.sql")
_COMPACT_MARK_MESSAGE_SQL = _load_sql("compact_mark_message.sql")
_LIST_SESSIONS_SQL = _load_sql("list_sessions.sql")
_DELETE_SESSION_SQL = _load_sql("delete_session.sql")
_DELETE_OLD_SESSIONS_SQL = _load_sql("delete_old_sessions.sql")
_DELETE_EXCESS_SESSIONS_SQL = _load_sql("delete_excess_sessions.sql")
_SEARCH_HISTORY_SQL = _load_sql("search_history.sql")
_SESSION_HISTORY_SQL = _load_sql("session_history.sql")
_GET_CREATED_AT_SQL = _load_sql("get_created_at.sql")
_COMPLETE_MESSAGE_SQL = _load_sql("complete_message.sql")
_FIND_LATEST_SUMMARY_SQL = _load_sql("find_latest_summary.sql")
_RESET_COMPACTION_SQL = _load_sql("reset_compaction.sql")
_HAS_MESSAGES_AFTER_SQL = _load_sql("has_messages_after.sql")
_DELETE_MESSAGE_SQL = _load_sql("delete_message.sql")
_SESSION_STARTED_AT_SQL = _load_sql("session_started_at.sql")


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Lightweight session listing — no message bodies."""

    session_id: str
    session_label: str | None
    last_active: str
    message_count: int
    total_cost: float
    model_name: str | None
    review_target: str | None
    repo_owner: str | None
    repo_name: str | None


# ── ResponseWriter ───────────────────────────────────────────────────


class ResponseWriter:
    """Incrementally persist a streaming model response.

    Created by :meth:`SessionStore.begin_response`.  The engine
    iterates PydanticAI stream events and calls :meth:`add_part`
    / :meth:`finish_part`.  The response is invisible to
    ``load_messages()`` until :meth:`finish` (or context-manager
    exit) sets ``status = 'complete'``.

    Usage::

        writer = store.begin_response(session_id, model_name="openai/gpt-4o")
        writer.add_part(0, initial_part)
        writer.finish_part(0, final_part)
        writer.finish(cost=0.003)
    """

    def __init__(
        self,
        *,
        store: SessionStore,
        message_id: str,
    ) -> None:
        self._store = store
        self.message_id = message_id
        self._part_ids: dict[int, str] = {}

    def add_part(self, index: int, part: ModelResponsePart) -> None:
        """Insert an in-progress content fragment for *part* at *index*.

        Called on ``PartStartEvent``.  The fragment has
        ``status = 'in_progress'`` until :meth:`finish_part` is called.
        """
        row_id = self._store.new_id()
        row = prepare_part_row(
            part,
            message_id=self.message_id,
            fragment_index=index + 1,
            context=self._store._ctx,
            row_id=row_id,
            status=FragmentStatus.IN_PROGRESS,
        )
        with self._store._lock, self._store._con:
            self._store._con.execute(_INSERT_FRAGMENT_SQL, dataclasses.astuple(row))
        self._part_ids[index] = row_id

    def finish_part(self, index: int, part: ModelResponsePart) -> None:
        """Update the fragment at *index* with the final part data.

        Called on ``PartEndEvent``.
        """
        fid = self._part_ids.get(index)
        if fid is None:
            return
        with self._store._lock, self._store._con:
            self._store._con.execute(_UPDATE_FRAGMENT_SQL, [dump_part(part), fid])

    def finish(
        self,
        *,
        cost: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_write_tokens: int | None = None,
    ) -> None:
        """Set ``status = 'complete'`` and record final metadata.

        Safe to call multiple times — the UPDATE is idempotent.
        A later call with ``cost`` / token counts overwrites earlier values.
        """
        with self._store._lock, self._store._con:
            self._store._con.execute(
                _COMPLETE_MESSAGE_SQL,
                [
                    cost,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_write_tokens,
                    self.message_id,
                ],
            )

    def __enter__(self) -> ResponseWriter:
        return self

    def __exit__(self, *args: object) -> None:
        # Auto-finish without cost on normal exit.  Callers that
        # know the cost should call finish() explicitly before exit.
        # Double-finish is safe — UPDATE is idempotent.
        self.finish()


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
        self._ctx = SessionContext(session_id="")
        self._setup()

    # ── Setup & migrations ───────────────────────────────────────────

    def _setup(self) -> None:
        """Apply schema and run migrations if needed."""
        self._con.execute("PRAGMA journal_mode = WAL")
        self._con.execute("PRAGMA synchronous = NORMAL")
        self._con.execute("PRAGMA foreign_keys = ON")

        version = self._user_version()
        if version == 0:
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

    def _migrate(self, from_version: int) -> None:
        """Run migrations from *from_version* to ``_SCHEMA_VERSION``.

        Databases older than the current date-based version are
        wiped and recreated — no incremental migration path.
        """
        log.info("sessions: wiping v%d DB, recreating as v%d", from_version, _SCHEMA_VERSION)
        self._wipe_and_recreate()

    def _user_version(self) -> int:
        row = self._con.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def _set_user_version(self, version: int) -> None:
        self._con.execute(f"PRAGMA user_version = {version}")

    def _wipe_and_recreate(self) -> None:
        tables = [
            row[0]
            for row in self._con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        for table in tables:
            self._con.execute(f"DROP TABLE IF EXISTS [{table}]")
        self._con.executescript(_SCHEMA_SQL)
        self._set_user_version(_SCHEMA_VERSION)

    # ── Public helpers ───────────────────────────────────────────────

    def new_id(self) -> str:
        """Generate a new UUID7 identifier (for sessions or rows)."""
        return str(uuid7())

    def set_context(
        self,
        session_id: str,
        *,
        session_label: str | None = None,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        model_name: str | None = None,
        review_target: str | None = None,
    ) -> None:
        """Set the session context used by all subsequent writes.

        Call once when the session starts or when metadata changes
        (e.g. model switch).  Thread-safe.
        """
        self._ctx = SessionContext(
            session_id=session_id,
            session_label=session_label,
            repo_owner=repo_owner,
            repo_name=repo_name,
            model_name=model_name,
            review_target=review_target,
        )

    # ── Writes ───────────────────────────────────────────────────────

    def save_messages(
        self,
        session_id: str,
        messages: list[ModelMessage],
        *,
        session_label: str | None = None,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        model_name: str | None = None,
        cost: float | None = None,
        status: FragmentStatus = FragmentStatus.COMPLETE,
    ) -> list[str]:
        """Persist a list of ``ModelMessage`` objects.

        Serialisation is handled internally.  *cost* is attributed to
        the last ``ModelResponse`` in *messages*.

        *status* sets the lifecycle state for all rows.  Use
        ``FragmentStatus.FAILED`` to persist a failed user prompt
        that should appear in ``search_history`` but not in
        ``load_messages``.

        Returns the list of message-level row IDs (one per message).

        When metadata kwargs are omitted, falls back to the context
        set by :meth:`set_context`.
        """
        if not messages:
            return []
        ctx = SessionContext(
            session_id=session_id,
            session_label=session_label or self._ctx.session_label,
            repo_owner=repo_owner or self._ctx.repo_owner,
            repo_name=repo_name or self._ctx.repo_name,
            model_name=model_name or self._ctx.model_name,
            review_target=self._ctx.review_target,
        )
        rows: list[Fragment] = []
        message_ids: list[str] = []
        last_response_idx: int | None = None

        for msg in messages:
            row_id = self.new_id()
            message_ids.append(row_id)
            rows.append(prepare_message_row(msg, context=ctx, row_id=row_id, status=status))
            rows.extend(prepare_part_rows(msg, message_id=row_id, context=ctx, status=status))
            if isinstance(msg, ModelResponse):
                last_response_idx = len(rows) - len(msg.parts) - 1

        # Set cost on the last response's message row at build time.
        if last_response_idx is not None and cost is not None:
            rows[last_response_idx] = dataclasses.replace(rows[last_response_idx], cost=cost)

        with self._lock, self._con:
            self._con.executemany(
                _INSERT_FRAGMENT_SQL,
                [dataclasses.astuple(r) for r in rows],
            )

        return message_ids

    def save_input(
        self,
        session_id: str,
        text: str,
        kind: str,
        *,
        session_label: str | None = None,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Persist a command or shell input as a self-referencing row.

        *kind* must be ``"command"`` or ``"shell"``.

        The row has ``message_id = id``, ``fragment_index = 0``,
        ``user_text = text``, no ``data_json``.  Visible in
        ``search_history`` but excluded from ``load_messages``.

        When metadata kwargs are omitted, falls back to the context
        set by :meth:`set_context`.
        """
        fk = FragmentKind(kind)
        row_id = self.new_id()
        ctx = SessionContext(
            session_id=session_id,
            session_label=session_label or self._ctx.session_label,
            repo_owner=repo_owner or self._ctx.repo_owner,
            repo_name=repo_name or self._ctx.repo_name,
            model_name=model_name or self._ctx.model_name,
            review_target=self._ctx.review_target,
        )
        row = prepare_input_row(text, fk, context=ctx, row_id=row_id)
        with self._lock, self._con:
            self._con.execute(_INSERT_FRAGMENT_SQL, dataclasses.astuple(row))

    def save_incident(
        self,
        session_id: str,
        kind: FragmentKind,
        payload: Incident,
    ) -> str:
        """Persist an incident (failure or history repair).

        *payload* is serialised via ``prepare_incident_row`` into
        ``data_json``.  Creates a self-referencing row
        (``message_id = id``, ``fragment_index = 0``).  Returns
        the row ID.

        Incident rows are excluded from ``load_messages`` (the SQL
        only selects ``request-message`` and ``response-message``
        kinds with ``status = 'complete'``).
        """
        row_id = self.new_id()
        row = prepare_incident_row(kind, payload, context=self._ctx, row_id=row_id)
        with self._lock, self._con:
            self._con.execute(_INSERT_FRAGMENT_SQL, dataclasses.astuple(row))
        return row_id

    def update_incident_json(self, row_id: str, key: str, value: str) -> None:
        """Set a top-level key in an incident row's ``data_json``.

        Uses SQLite's ``json_set`` for an atomic single-statement
        update — no read-modify-write.
        """
        with self._lock, self._con:
            self._con.execute(
                "UPDATE fragments SET data_json = json_set(data_json, '$.' || ?, ?) WHERE id = ?",
                [key, value, row_id],
            )

    # ── Streaming writes ────────────────────────────────────────────

    def begin_response(
        self,
        session_id: str,
        *,
        model_name: str | None = None,
    ) -> ResponseWriter:
        """Start streaming a model response.

        Inserts an in-progress message header (``status = 'in_progress'``) and
        returns a :class:`ResponseWriter` for adding parts
        incrementally.  The message is invisible to
        ``load_messages()`` until the writer is finished.
        """
        row_id = self.new_id()
        placeholder = ModelResponse(parts=[], model_name=model_name)
        row = prepare_message_row(
            placeholder, context=self._ctx, row_id=row_id, status=FragmentStatus.IN_PROGRESS
        )
        with self._lock, self._con:
            self._con.execute(_INSERT_FRAGMENT_SQL, dataclasses.astuple(row))
        return ResponseWriter(store=self, message_id=row_id)

    # ── Compaction ───────────────────────────────────────────────────

    def compact_session(
        self,
        session_id: str,
        *,
        summary: ModelMessage,
        compact_ids: list[str],
        session_label: str | None = None,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Compact a session: insert summary, mark old messages.

        1. Insert the summary message + its content fragments.
        2. Mark each message in *compact_ids* (and its fragments)
           as compacted by the summary.

        Kept messages are **not re-inserted** — they remain as-is
        in the DB.  Only the messages in *compact_ids* are marked.
        The summary inherits the ``created_at`` of the earliest
        compacted message so it sorts before the kept messages.

        When metadata kwargs are omitted, falls back to the context
        set by :meth:`set_context`.
        """
        if not compact_ids:
            return

        ctx = SessionContext(
            session_id=session_id,
            session_label=session_label or self._ctx.session_label,
            repo_owner=repo_owner or self._ctx.repo_owner,
            repo_name=repo_name or self._ctx.repo_name,
            model_name=model_name or self._ctx.model_name,
        )

        summary_id = self.new_id()
        summary_rows: list[Fragment] = [
            prepare_message_row(summary, context=ctx, row_id=summary_id),
            *prepare_part_rows(summary, message_id=summary_id, context=ctx),
        ]

        # Use the earliest compacted message's created_at for the
        # summary so it sorts before the kept messages.
        first_id = compact_ids[0]
        row = self._con.execute(_GET_CREATED_AT_SQL, [first_id]).fetchone()
        if row is not None:
            earliest = row["created_at"]
            summary_rows = [dataclasses.replace(r, created_at=earliest) for r in summary_rows]

        with self._lock, self._con:
            self._con.executemany(
                _INSERT_FRAGMENT_SQL,
                [dataclasses.astuple(r) for r in summary_rows],
            )
            self._con.executemany(
                _COMPACT_MARK_MESSAGE_SQL,
                [[summary_id, mid, mid] for mid in compact_ids],
            )

    def reset_latest_compaction(self, session_id: str) -> int:
        """Undo the most recent compaction for a session.

        Sets ``compacted_by = NULL`` on all fragments that were
        marked by the most recent summary, then deletes the summary
        message.  The summary is removed because its timestamp
        would interleave with restored messages and break tool-call
        pairing.

        Returns the number of fragments restored, 0 if none existed.

        Raises :class:`ValueError` if messages were added after the
        compaction (the summary is already part of later context).
        """
        with self._lock, self._con:
            row = self._con.execute(_FIND_LATEST_SUMMARY_SQL, [session_id]).fetchone()
            if row is None:
                return 0
            summary_id = row["summary_id"]
            after = self._con.execute(_HAS_MESSAGES_AFTER_SQL, [session_id, summary_id]).fetchone()
            if after and after["new_count"] > 0:
                raise ValueError(
                    "Cannot reset — messages were added after this compaction. "
                    "The summary is already part of the conversation context."
                )
            cur = self._con.execute(_RESET_COMPACTION_SQL, [summary_id])
            restored = cur.rowcount
            # Delete the summary — its timestamp would interleave
            # with restored messages and break tool-call ordering.
            self._con.execute(_DELETE_MESSAGE_SQL, [summary_id])
            return restored

    def delete_session(self, session_id: str) -> int:
        """Delete all fragments for a session.  Returns rows deleted."""
        with self._lock, self._con:
            cur = self._con.execute(_DELETE_SESSION_SQL, [session_id])
            return cur.rowcount

    def delete_old_sessions(self, before: datetime) -> int:
        """Delete sessions whose last activity is before *before*."""
        with self._lock, self._con:
            cur = self._con.execute(_DELETE_OLD_SESSIONS_SQL, [before.isoformat()])
            return cur.rowcount

    def delete_excess_sessions(self, keep: int) -> int:
        """Keep only the *keep* most recent sessions, delete the rest."""
        if keep < 1:
            return 0
        with self._lock, self._con:
            cur = self._con.execute(_DELETE_EXCESS_SESSIONS_SQL, [keep])
            return cur.rowcount

    # ── Reads ────────────────────────────────────────────────────────

    def load_messages(self, session_id: str) -> list[ModelMessage]:
        """Load active messages (``status = 'complete'``) for a session.

        Reconstructs ``ModelMessage`` objects from fragment rows.
        Returns ``list[ModelMessage]`` in chronological order.
        Corrupt rows are logged and skipped.
        """
        return [msg for _id, msg in self._load_messages_paired(session_id)]

    def load_messages_with_ids(self, session_id: str) -> list[tuple[str, ModelMessage]]:
        """Load messages paired with their DB row IDs.

        Same as :meth:`load_messages` but returns
        ``list[(message_id, ModelMessage)]``.  Used by compaction
        to get IDs from the same query that loads messages —
        no index-alignment with a separate query.
        """
        return self._load_messages_paired(session_id)

    def _load_messages_paired(self, session_id: str) -> list[tuple[str, ModelMessage]]:
        rows = self._con.execute(_LOAD_MESSAGES_SQL, [session_id]).fetchall()
        if not rows:
            return []

        from itertools import groupby

        result: list[tuple[str, ModelMessage]] = []

        for mid, group in groupby(rows, key=lambda r: r["message_id"]):
            fragment_list = list(group)
            message_row = None
            part_jsons: list[str] = []
            for r in fragment_list:
                if r["id"] == r["message_id"] and r["fragment_index"] == 0:
                    message_row = r
                elif r["data_json"]:
                    part_jsons.append(r["data_json"])

            if message_row is None:
                log.warning("sessions: orphan fragments for message %s", mid)
                continue

            fk = FragmentKind(message_row["fragment_kind"])
            if not fk.is_message:
                continue

            try:
                msg = reconstruct_message(fk, message_row["data_json"], part_jsons)
                result.append((mid, msg))
            except (ValidationError, ValueError, KeyError):
                log.warning("sessions: corrupt message %s", mid)

        return result

    def load_message_ids(
        self,
        session_id: str,
        *,
        before_created_at: str | None = None,
    ) -> list[str]:
        """Return message-level row IDs for a session, optionally filtered."""
        rows = self._con.execute(
            _LOAD_MESSAGE_IDS_SQL,
            [session_id, before_created_at, before_created_at],
        ).fetchall()
        return [r["id"] for r in rows]

    def list_sessions(
        self,
        *,
        repo_owner: str | None = None,
        repo_name: str | None = None,
        limit: int = 50,
    ) -> list[SessionSummary]:
        """List sessions, most recent first."""
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
                model_name=row["model_name"],
                review_target=row["review_target"],
                repo_owner=row["repo_owner"],
                repo_name=row["repo_name"],
            )
            for row in rows
        ]

    def session_history(self, session_id: str, limit: int = 10) -> list[str]:
        """Return the most recent user inputs for a session (newest first)."""
        rows = self._con.execute(
            _SESSION_HISTORY_SQL,
            [session_id, limit],
        ).fetchall()
        return [row["user_text"] for row in rows]

    def search_history(
        self,
        prefix: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """Search user input history across all sessions."""
        rows = self._con.execute(
            _SEARCH_HISTORY_SQL,
            [prefix, prefix, limit],
        ).fetchall()
        return [row["user_text"] for row in rows]

    def session_started_at(self, session_id: str) -> float | None:
        """Return the wall-clock timestamp of the session's first message.

        Returns ``None`` if the session has no messages.
        """
        row = self._con.execute(_SESSION_STARTED_AT_SQL, [session_id]).fetchone()
        if row is None or row["started"] is None:
            return None
        dt = datetime.fromisoformat(row["started"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()

    def token_stats(self, session_id: str) -> TokenStats:
        """Return token usage for a session, split by compaction status."""
        return _token_stats(self._con, session_id)

    def tool_stats(self, session_id: str) -> list[ToolStat]:
        """Return per-tool call and failure counts for a session."""
        return _tool_stats(self._con, session_id)

    def incident_stats(self, session_id: str) -> IncidentStats:
        """Return failure and repair incident stats for a session."""
        return _incident_stats(self._con, session_id)

    def global_stats(self) -> GlobalStats:
        """Return aggregate statistics across all persisted sessions."""
        return _global_stats(self._con)

    def global_incident_stats(self) -> IncidentStats:
        """Return failure and repair incident stats across all sessions."""
        return _global_incident_stats(self._con)

    def update_session_label(self, session_id: str, label: str) -> int:
        """Update the label on all fragments for a session.

        Returns the number of rows updated.
        """
        with self._lock, self._con:
            cur = self._con.execute(
                "UPDATE fragments SET session_label = ? WHERE session_id = ?",
                [label, session_id],
            )
            return cur.rowcount

    # ── Lifecycle ────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._con.close()

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
