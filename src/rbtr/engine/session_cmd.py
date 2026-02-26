"""Handler for /session — list, inspect, delete, purge, resume."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from rbtr.sessions.store import SessionSummary
from rbtr.styles import STYLE_DIM

if TYPE_CHECKING:
    from .core import Engine

# ── Duration parsing ─────────────────────────────────────────────────

_UNIT_MAP: dict[str, str] = {
    "d": "days",
    "w": "weeks",
    "h": "hours",
}


def _parse_duration(spec: str) -> timedelta | None:
    """Parse a duration like ``7d``, ``2w``, ``24h``.

    Returns ``None`` if the format is not recognised.
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
  /session resume <id> Resume a previous session (prefix match)
  /session delete <id> Delete a session by ID (prefix match)
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
        case "resume":
            _cmd_resume(engine, rest)
        case "resume-last":
            _cmd_resume_last(engine)
        case "delete":
            _cmd_delete(engine, rest)
        case "purge":
            _cmd_purge(engine, rest)
        case "help":
            engine._out(_HELP, style=STYLE_DIM)
        case _:
            engine._warn(f"Unknown subcommand: {subcmd}")
            engine._out(_HELP, style=STYLE_DIM)


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
    _render_session_list(engine, sessions)


def _render_session_list(engine: Engine, sessions: list[SessionSummary]) -> None:
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
        engine._out(
            f"  {short_id}  {age:>6}  {s.message_count:>4} msgs  {cost:>10}  {label}{marker}",
            style=STYLE_DIM,
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
    engine._out(f"  Session ID    {s.session_id[:8]}", style=STYLE_DIM)
    engine._out(f"  Label         {s.session_label or '—'}", style=STYLE_DIM)
    engine._out(f"  Repo          {repo}", style=STYLE_DIM)
    engine._out(f"  Model         {s.model_name or '—'}", style=STYLE_DIM)
    msg_count = len(engine.store.load_messages(s.session_id))
    engine._out(f"  Messages      {msg_count}", style=STYLE_DIM)


# ── resume ───────────────────────────────────────────────────────────


def _cmd_resume(engine: Engine, args: list[str]) -> None:
    """Resume a previous session by ID prefix."""
    if not args:
        engine._warn("Usage: /session resume <id>")
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
            engine._out(f"  {s.session_id[:12]}  {s.session_label or '—'}", style=STYLE_DIM)
        return

    target = matches[0]
    if target.session_id == engine.state.session_id:
        engine._warn("Already in this session.")
        return

    messages = engine.store.load_messages(target.session_id)
    if not messages:
        engine._warn("Session has no messages (may have been compacted).")
        return

    # Switch session and restore usage counters from DB.
    engine.state.session_id = target.session_id
    engine.state.session_label = target.session_label or engine.state.session_label

    ts = engine.store.token_stats(target.session_id)
    engine.state.usage.restore(
        turn_count=ts.total_turns,
        response_count=ts.total_responses,
        input_tokens=ts.total_input_tokens,
        output_tokens=ts.total_output_tokens,
        cost=ts.total_cost,
    )

    label = target.session_label or target.session_id[:8]
    engine._out(f"Resumed session '{label}' ({len(messages)} messages).")

    # Restore the review target (re-fetches PR metadata / rebuilds index).
    if target.review_target:
        from .review import cmd_review

        engine._out(f"Restoring review target: /review {target.review_target}")
        cmd_review(engine, target.review_target)


def _cmd_resume_last(engine: Engine) -> None:
    """Resume the most recent session for the current repo.

    Used by ``rbtr -c`` at startup.  Silently does nothing when
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
            engine._out(f"  {s.session_id[:12]}  {s.session_label or '—'}", style=STYLE_DIM)
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

    duration = _parse_duration(args[0])
    if duration is None:
        engine._warn(f"Invalid duration: {args[0]}. Use e.g. 7d, 2w, 24h.")
        return

    cutoff = datetime.now(UTC) - duration
    deleted = engine.store.delete_old_sessions(before=cutoff)
    engine._out(f"Deleted {deleted} message{'s' if deleted != 1 else ''} from old sessions.")
