"""Handler for /session — list, inspect, delete, purge, resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from rbtr.sessions.store import SessionSummary

from .review_cmd import cmd_review

if TYPE_CHECKING:
    from .core import Engine

# ── Duration parsing ─────────────────────────────────────────────────

_UNIT_MAP: dict[str, str] = {
    "d": "days",
    "w": "weeks",
    "h": "hours",
}


def parse_duration(spec: str) -> timedelta | None:
    """Parse a duration like `7d`, `2w`, `24h`.

    Returns `None` if the format is not recognised.
    """
    if len(spec) < 2:
        return None
    unit = _UNIT_MAP.get(spec[-1])
    if unit is None:
        return None
    try:
        value = int(spec[:-1])
    except ValueError:
        return None
    return timedelta(**{unit: value})


# ── Subcommands ──────────────────────────────────────────────────────

_HELP = """\
  /session             List recent sessions (current repo)
  /session all         List sessions across all repos
  /session info        Show current session details
  /session rename <n>  Rename the current session
  /session history     Show the last 10 inputs in this session
  /session resume <q>  Resume a session (ID prefix or label)
  /session delete <id> Delete a session by ID prefix
  /session purge <dur> Delete sessions older than duration (e.g. 7d, 2w)\
"""


def cmd_session(engine: Engine, args: str) -> None:
    """Dispatch /session subcommands."""
    parts = args.split()
    subcmd = parts[0] if parts else ""
    rest = parts[1:]

    match subcmd:
        case "" | "list":
            _cmd_list(engine)
        case "all":
            _cmd_all(engine)
        case "info":
            _cmd_info(engine)
        case "rename":
            _cmd_rename(engine, rest)
        case "resume":
            _cmd_resume(engine, rest)
        case "resume-last":
            _cmd_resume_last(engine)
        case "history":
            _cmd_history(engine)
        case "delete":
            _cmd_delete(engine, rest)
        case "purge":
            _cmd_purge(engine, rest)
        case "help":
            engine._out(_HELP)
        case _:
            engine._warn(f"Unknown subcommand: {subcmd}")
            engine._out(_HELP)


# ── list / all ───────────────────────────────────────────────────────


def _cmd_list(engine: Engine) -> None:
    """List recent sessions for the current repo."""
    sessions = engine.store.list_sessions(
        repo_owner=engine.state.owner or None,
        repo_name=engine.state.repo_name or None,
    )
    _render_session_list(engine, sessions)


def _cmd_all(engine: Engine) -> None:
    """List sessions across all repos."""
    sessions = engine.store.list_sessions()
    _render_session_list(engine, sessions, show_repo=True)


def _render_session_list(
    engine: Engine,
    sessions: list[SessionSummary],
    *,
    show_repo: bool = False,
) -> None:
    """Render a session listing."""
    if not sessions:
        engine._out("No sessions found.")
        return

    current_id = engine.state.session_id
    for s in sessions:
        short_id = s.session_id[:8]
        label = s.session_label or "—"
        cost = f"${s.total_cost:.4f}" if s.total_cost else "—"
        marker = " ◂" if s.session_id == current_id else ""
        age = _format_age(s.last_active)
        repo = ""
        if show_repo and s.repo_owner and s.repo_name:
            repo = f"  {s.repo_owner}/{s.repo_name}"
        engine._out(
            f"  {short_id}  {age:>6}  {s.message_count:>4} msgs  {cost:>10}{repo}  {label}{marker}",
        )


def _format_age(iso_timestamp: str) -> str:
    """Format an ISO timestamp as a human-readable relative age."""
    try:
        ts = datetime.fromisoformat(iso_timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - ts
    except (ValueError, TypeError):
        return "?"

    if delta.days > 365:
        return f"{delta.days // 365}y"
    if delta.days > 30:
        return f"{delta.days // 30}mo"
    if delta.days > 0:
        return f"{delta.days}d"
    hours = delta.seconds // 3600
    if hours > 0:
        return f"{hours}h"
    minutes = delta.seconds // 60
    if minutes > 0:
        return f"{minutes}m"
    return "now"


# ── info ─────────────────────────────────────────────────────────────


def _cmd_info(engine: Engine) -> None:
    """Show current session details."""
    s = engine.state
    repo = f"{s.owner}/{s.repo_name}" if s.owner else "—"
    engine._out(f"  Session ID    {s.session_id[:8]}")
    engine._out(f"  Label         {s.session_label or '—'}")
    engine._out(f"  Repo          {repo}")
    engine._out(f"  Model         {s.model_name or '—'}")
    ts = engine.store.token_stats(s.session_id)
    engine._out(f"  Turns         {ts.active_turns}")
    engine._out(f"  Responses     {ts.active_responses}")


# ── history ──────────────────────────────────────────────────────────

_HISTORY_LIMIT = 10


def _cmd_history(engine: Engine) -> None:
    """Show the last user inputs in the current session."""
    # Fetch one extra — the most recent entry is this command itself.
    entries = engine.store.session_history(engine.state.session_id, _HISTORY_LIMIT + 1)
    entries = [e for e in entries if e != "/session history"]
    if not entries:
        engine._out("No inputs in this session yet.")
        return
    # Entries are newest-first from the query; display oldest-first.
    for i, text in enumerate(reversed(entries), 1):
        # Truncate long inputs to a single line for readability.
        preview = text.replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "…"
        engine._out(f"  {i:>2}. {preview}")


# ── rename ───────────────────────────────────────────────────────────


def _cmd_rename(engine: Engine, args: list[str]) -> None:
    """Rename the current session."""
    if not args:
        engine._warn("Usage: /session rename <name>")
        return

    label = " ".join(args)
    engine.state.session_label = label
    engine.store.update_session_label(engine.state.session_id, label)
    engine._out(f"Session renamed to '{label}'.")
    engine._context(f"[/session rename → {label}]", f"Renamed session to '{label}'.")


# ── Session lookup ───────────────────────────────────────────────────


def _find_session(engine: Engine, query: str) -> SessionSummary | None:
    """Find a session by ID prefix or label substring.

    Tries ID prefix first.  Falls back to case-insensitive label
    substring — when several sessions share a label the most
    recent one wins (`list_sessions` returns newest first).
    Returns `None` (with a user-facing warning) on zero or
    ambiguous ID-prefix matches.
    """
    sessions = engine.store.list_sessions(limit=200)

    # Try ID prefix first — must be unambiguous.
    by_id = [s for s in sessions if s.session_id.startswith(query)]
    if len(by_id) == 1:
        return by_id[0]
    if len(by_id) > 1:
        engine._warn(f"Ambiguous ID prefix '{query}' — matches {len(by_id)} sessions.")
        for s in by_id[:5]:
            engine._out(f"  {s.session_id[:12]}  {s.session_label or '—'}")
        return None

    # Fall back to label substring (case-insensitive).
    # Most recent match wins — no ambiguity error for duplicate labels.
    lower = query.lower()
    for s in sessions:
        if s.session_label and lower in s.session_label.lower():
            return s

    engine._warn(f"No session matching '{query}'.")
    return None


# ── resume ───────────────────────────────────────────────────────────


def _cmd_resume(engine: Engine, args: list[str]) -> None:
    """Resume a previous session by ID prefix or label substring."""
    if not args:
        engine._warn("Usage: /session resume <id or label>")
        return

    query = " ".join(args)
    target = _find_session(engine, query)
    if target is None:
        return

    if target.session_id == engine.state.session_id:
        engine._warn("Already in this session.")
        return

    messages = engine.store.load_messages(target.session_id)
    if not messages:
        engine._warn("Session has no messages (may have been compacted).")
        return

    # Switch session and restore usage counters from DB.
    engine.state.session_id = target.session_id
    engine.state.session_label = target.session_label or ""

    # Restore session start time from the earliest message.
    started = engine.store.session_started_at(target.session_id)
    if started is not None:
        engine.state.session_started_at = started

    ts = engine.store.token_stats(target.session_id)
    oh = engine.store.overhead_stats(target.session_id)
    engine.state.usage.restore(
        turn_count=ts.active_turns,
        response_count=ts.active_responses,
        input_tokens=ts.active_input_tokens,
        output_tokens=ts.active_output_tokens,
        cost=ts.active_cost,
        compaction_input_tokens=oh.compaction_input_tokens,
        compaction_output_tokens=oh.compaction_output_tokens,
        compaction_cost=oh.compaction_cost,
        fact_extraction_input_tokens=oh.fact_extraction_input_tokens,
        fact_extraction_output_tokens=oh.fact_extraction_output_tokens,
        fact_extraction_cost=oh.fact_extraction_cost,
    )

    label = target.session_label or target.session_id[:8]
    engine._out(f"Resumed session '{label}' ({ts.active_turns} turns).")

    # Restore the review target (re-fetches PR metadata / rebuilds index).
    # Skip when the session belongs to a different repo — the target
    # (PR number or branch name) would be meaningless here.
    if target.review_target:
        same_repo = (
            target.repo_owner == engine.state.owner and target.repo_name == engine.state.repo_name
        )
        if same_repo:
            engine._out(f"Restoring review target: /review {target.review_target}")
            cmd_review(engine, target.review_target)
        else:
            engine._warn(
                f"Review target '{target.review_target}' belongs to"
                f" {target.repo_owner}/{target.repo_name} — skipped."
            )


def _cmd_resume_last(engine: Engine) -> None:
    """Resume the most recent session for the current repo.

    Used by `rbtr -c` at startup.  Silently does nothing when
    there is no previous session to resume.
    """
    sessions = engine.store.list_sessions(
        repo_owner=engine.state.owner,
        repo_name=engine.state.repo_name,
        limit=1,
    )
    if not sessions:
        engine._out("No previous session for this repo.")
        return
    _cmd_resume(engine, [sessions[0].session_id])


# ── delete ───────────────────────────────────────────────────────────


def _cmd_delete(engine: Engine, args: list[str]) -> None:
    """Delete a session by ID prefix."""
    if not args:
        engine._warn("Usage: /session delete <id>")
        return

    prefix = args[0]
    sessions = engine.store.list_sessions(limit=200)
    matches = [s for s in sessions if s.session_id.startswith(prefix)]

    if not matches:
        engine._warn(f"No session matching '{prefix}'.")
        return
    if len(matches) > 1:
        engine._warn(f"Ambiguous prefix '{prefix}' — matches {len(matches)} sessions.")
        for s in matches[:5]:
            engine._out(f"  {s.session_id[:12]}  {s.session_label or '—'}")
        return

    target = matches[0]
    if target.session_id == engine.state.session_id:
        engine._warn("Cannot delete the active session. Use /new first.")
        return

    deleted = engine.store.delete_session(target.session_id)
    label = target.session_label or target.session_id[:8]
    engine._out(f"Deleted session '{label}' ({deleted} messages).")


# ── purge ────────────────────────────────────────────────────────────


def _cmd_purge(engine: Engine, args: list[str]) -> None:
    """Delete sessions older than a duration."""
    if not args:
        engine._warn("Usage: /session purge <duration> (e.g. 7d, 2w, 24h)")
        return

    duration = parse_duration(args[0])
    if duration is None:
        engine._warn(f"Invalid duration: {args[0]}. Use e.g. 7d, 2w, 24h.")
        return

    cutoff = datetime.now(UTC) - duration
    deleted = engine.store.delete_old_sessions(before=cutoff)
    engine._out(f"Deleted {deleted} message{'s' if deleted != 1 else ''} from old sessions.")
