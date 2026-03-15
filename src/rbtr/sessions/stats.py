"""Read-only session analytics — token usage and tool frequency.

Pure query functions over the `fragments` table.  Each takes a
`sqlite3.Connection` and returns frozen dataclasses.  The
`SessionStore` delegates to these via
thin wrapper methods.
"""

from __future__ import annotations

import importlib.resources
import sqlite3
from dataclasses import dataclass

# ── SQL loader ───────────────────────────────────────────────────────

_sql_pkg = importlib.resources.files("rbtr.sessions") / "sql"


def _load_sql(name: str) -> str:
    return (_sql_pkg / name).read_text(encoding="utf-8")


_SESSION_TOKEN_STATS_SQL = _load_sql("session_token_stats.sql")
_TOOL_STATS_SQL = _load_sql("tool_stats.sql")
_GLOBAL_STATS_SQL = _load_sql("global_stats.sql")
_GLOBAL_MODEL_STATS_SQL = _load_sql("global_model_stats.sql")
_GLOBAL_TOOL_STATS_SQL = _load_sql("global_tool_stats.sql")
_SESSION_OVERHEAD_STATS_SQL = _load_sql("session_overhead_stats.sql")
_GLOBAL_OVERHEAD_STATS_SQL = _load_sql("global_overhead_stats.sql")
_INCIDENT_FAILURE_STATS_SQL = _load_sql("incident_failure_stats.sql")
_INCIDENT_REPAIR_STATS_SQL = _load_sql("incident_repair_stats.sql")
_GLOBAL_INCIDENT_FAILURE_STATS_SQL = _load_sql("global_incident_failure_stats.sql")
_GLOBAL_INCIDENT_REPAIR_STATS_SQL = _load_sql("global_incident_repair_stats.sql")


# ── Result types ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ToolStat:
    """Call and failure counts for a single tool in a session."""

    tool_name: str
    call_count: int
    active_call_count: int
    failure_count: int


@dataclass(frozen=True, slots=True)
class TokenStats:
    """Token usage for a session, split by compaction status.

    `total_*` = lifetime (all messages ever).
    `active_*` = non-compacted messages only.
    Compacted = `total - active`.
    """

    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    total_cost: float
    active_input_tokens: int
    active_output_tokens: int
    active_cache_read_tokens: int
    active_cache_write_tokens: int
    active_cost: float
    total_responses: int
    active_responses: int
    total_turns: int
    active_turns: int
    compaction_count: int


_EMPTY_TOKEN_STATS = TokenStats(
    total_input_tokens=0,
    total_output_tokens=0,
    total_cache_read_tokens=0,
    total_cache_write_tokens=0,
    total_cost=0.0,
    active_input_tokens=0,
    active_output_tokens=0,
    active_cache_read_tokens=0,
    active_cache_write_tokens=0,
    active_cost=0.0,
    total_responses=0,
    active_responses=0,
    total_turns=0,
    active_turns=0,
    compaction_count=0,
)


# ── Query functions ──────────────────────────────────────────────────


def token_stats(con: sqlite3.Connection, session_id: str) -> TokenStats:
    """Return token usage for a session, split by compaction status."""
    row = con.execute(_SESSION_TOKEN_STATS_SQL, [session_id, session_id, session_id]).fetchone()
    if row is None or row["total_input_tokens"] is None:
        return _EMPTY_TOKEN_STATS
    return TokenStats(
        total_input_tokens=int(row["total_input_tokens"]),
        total_output_tokens=int(row["total_output_tokens"]),
        total_cache_read_tokens=int(row["total_cache_read_tokens"]),
        total_cache_write_tokens=int(row["total_cache_write_tokens"]),
        total_cost=float(row["total_cost"]),
        active_input_tokens=int(row["active_input_tokens"]),
        active_output_tokens=int(row["active_output_tokens"]),
        active_cache_read_tokens=int(row["active_cache_read_tokens"]),
        active_cache_write_tokens=int(row["active_cache_write_tokens"]),
        active_cost=float(row["active_cost"]),
        total_responses=int(row["total_responses"]),
        active_responses=int(row["active_responses"]),
        total_turns=int(row["total_turns"]),
        active_turns=int(row["active_turns"]),
        compaction_count=int(row["compaction_count"]),
    )


def tool_stats(con: sqlite3.Connection, session_id: str) -> list[ToolStat]:
    """Return per-tool call and failure counts for a session."""
    rows = con.execute(_TOOL_STATS_SQL, [session_id]).fetchall()
    return [
        ToolStat(
            tool_name=row["tool_name"],
            call_count=int(row["call_count"]),
            active_call_count=int(row["active_call_count"]),
            failure_count=int(row["failure_count"]),
        )
        for row in rows
    ]


# ── Incident stats ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FailureStat:
    """Per-failure-kind counts for a session."""

    failure_kind: str
    total_count: int
    recovered_count: int
    failed_count: int


@dataclass(frozen=True, slots=True)
class RepairStat:
    """Per-strategy counts for history repairs in a session."""

    strategy: str
    total_count: int
    reason: str | None


@dataclass(frozen=True, slots=True)
class IncidentStats:
    """Failure and repair incident stats for a session.

    Empty lists when the session has no incidents.
    """

    failures: list[FailureStat]
    repairs: list[RepairStat]

    @property
    def has_incidents(self) -> bool:
        """True if any incidents exist."""
        return bool(self.failures or self.repairs)

    @property
    def total_failures(self) -> int:
        """Total failure count across all kinds."""
        return sum(f.total_count for f in self.failures)

    @property
    def total_recovered(self) -> int:
        """Total recovered count across all kinds."""
        return sum(f.recovered_count for f in self.failures)


_EMPTY_INCIDENT_STATS = IncidentStats(failures=[], repairs=[])


def incident_stats(con: sqlite3.Connection, session_id: str) -> IncidentStats:
    """Return failure and repair incident stats for a session."""
    failure_rows = con.execute(_INCIDENT_FAILURE_STATS_SQL, [session_id]).fetchall()
    repair_rows = con.execute(_INCIDENT_REPAIR_STATS_SQL, [session_id]).fetchall()

    if not failure_rows and not repair_rows:
        return _EMPTY_INCIDENT_STATS

    failures = [
        FailureStat(
            failure_kind=r["failure_kind"],
            total_count=int(r["total_count"]),
            recovered_count=int(r["recovered_count"]),
            failed_count=int(r["failed_count"]),
        )
        for r in failure_rows
    ]
    repairs = [
        RepairStat(
            strategy=r["strategy"],
            total_count=int(r["total_count"]),
            reason=r["reason"],
        )
        for r in repair_rows
    ]
    return IncidentStats(failures=failures, repairs=repairs)


def global_incident_stats(con: sqlite3.Connection) -> IncidentStats:
    """Return failure and repair incident stats across all sessions."""
    failure_rows = con.execute(_GLOBAL_INCIDENT_FAILURE_STATS_SQL).fetchall()
    repair_rows = con.execute(_GLOBAL_INCIDENT_REPAIR_STATS_SQL).fetchall()

    if not failure_rows and not repair_rows:
        return _EMPTY_INCIDENT_STATS

    failures = [
        FailureStat(
            failure_kind=r["failure_kind"],
            total_count=int(r["total_count"]),
            recovered_count=int(r["recovered_count"]),
            failed_count=int(r["failed_count"]),
        )
        for r in failure_rows
    ]
    repairs = [
        RepairStat(
            strategy=r["strategy"],
            total_count=int(r["total_count"]),
            reason=r["reason"],
        )
        for r in repair_rows
    ]
    return IncidentStats(failures=failures, repairs=repairs)


# ── Overhead stats ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OverheadStats:
    """Compaction and fact extraction cost breakdown for a session or globally."""

    compaction_input_tokens: int
    compaction_output_tokens: int
    compaction_cost: float
    compaction_count: int
    fact_extraction_input_tokens: int
    fact_extraction_output_tokens: int
    fact_extraction_cost: float
    fact_extraction_count: int

    @property
    def has_overhead(self) -> bool:
        """True if any overhead was recorded."""
        return self.compaction_count > 0 or self.fact_extraction_count > 0

    @property
    def total_cost(self) -> float:
        """Combined compaction + fact extraction cost."""
        return self.compaction_cost + self.fact_extraction_cost


_EMPTY_OVERHEAD_STATS = OverheadStats(
    compaction_input_tokens=0,
    compaction_output_tokens=0,
    compaction_cost=0.0,
    compaction_count=0,
    fact_extraction_input_tokens=0,
    fact_extraction_output_tokens=0,
    fact_extraction_cost=0.0,
    fact_extraction_count=0,
)


def _parse_overhead_row(row: sqlite3.Row | None) -> OverheadStats:
    """Build `OverheadStats` from a SQL result row."""
    if row is None or (row["compaction_count"] == 0 and row["fact_extraction_count"] == 0):
        return _EMPTY_OVERHEAD_STATS
    return OverheadStats(
        compaction_input_tokens=int(row["compaction_input_tokens"]),
        compaction_output_tokens=int(row["compaction_output_tokens"]),
        compaction_cost=float(row["compaction_cost"]),
        compaction_count=int(row["compaction_count"]),
        fact_extraction_input_tokens=int(row["fact_extraction_input_tokens"]),
        fact_extraction_output_tokens=int(row["fact_extraction_output_tokens"]),
        fact_extraction_cost=float(row["fact_extraction_cost"]),
        fact_extraction_count=int(row["fact_extraction_count"]),
    )


def overhead_stats(con: sqlite3.Connection, session_id: str) -> OverheadStats:
    """Return overhead cost stats for a session."""
    row = con.execute(_SESSION_OVERHEAD_STATS_SQL, [session_id]).fetchone()
    return _parse_overhead_row(row)


def global_overhead_stats(con: sqlite3.Connection) -> OverheadStats:
    """Return overhead cost stats across all sessions."""
    row = con.execute(_GLOBAL_OVERHEAD_STATS_SQL).fetchone()
    return _parse_overhead_row(row)


# ── Global (cross-session) stats ─────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelStat:
    """Per-model cost and token breakdown across all sessions."""

    model_name: str
    session_count: int
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int


@dataclass(frozen=True, slots=True)
class GlobalStats:
    """Aggregate statistics across all persisted sessions."""

    session_count: int
    total_cost: float
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    models: list[ModelStat]
    tools: list[ToolStat]


_EMPTY_GLOBAL_STATS = GlobalStats(
    session_count=0,
    total_cost=0.0,
    total_input_tokens=0,
    total_output_tokens=0,
    total_cache_read_tokens=0,
    total_cache_write_tokens=0,
    models=[],
    tools=[],
)


def global_stats(con: sqlite3.Connection) -> GlobalStats:
    """Return aggregate statistics across all persisted sessions."""
    # Totals.
    row = con.execute(_GLOBAL_STATS_SQL).fetchone()
    if row is None or row["session_count"] == 0:
        return _EMPTY_GLOBAL_STATS

    # Per-model breakdown.
    model_rows = con.execute(_GLOBAL_MODEL_STATS_SQL).fetchall()
    models = [
        ModelStat(
            model_name=r["model_name"],
            session_count=int(r["session_count"]),
            total_cost=float(r["total_cost"]),
            total_input_tokens=int(r["total_input_tokens"]),
            total_output_tokens=int(r["total_output_tokens"]),
        )
        for r in model_rows
    ]

    # Tool frequency.
    tool_rows = con.execute(_GLOBAL_TOOL_STATS_SQL).fetchall()
    tools = [
        ToolStat(
            tool_name=r["tool_name"],
            call_count=int(r["call_count"]),
            active_call_count=int(r["call_count"]),  # no compaction at global level
            failure_count=int(r["failure_count"]),
        )
        for r in tool_rows
    ]

    return GlobalStats(
        session_count=int(row["session_count"]),
        total_cost=float(row["total_cost"]),
        total_input_tokens=int(row["total_input_tokens"]),
        total_output_tokens=int(row["total_output_tokens"]),
        total_cache_read_tokens=int(row["total_cache_read_tokens"]),
        total_cache_write_tokens=int(row["total_cache_write_tokens"]),
        models=models,
        tools=tools,
    )
