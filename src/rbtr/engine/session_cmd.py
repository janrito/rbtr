"""Handler for /session — list, inspect, and delete sessions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

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
  /session list        Same as above
  /session list --all  List sessions across all repos
  /session info        Show current session details
  /session delete <id> Delete a session by ID (prefix match)
  /session delete --before <duration>  Delete sessions older than duration (e.g. 7d, 2w)\
"""


def cmd_session(engine: Engine, args: str) -> None:
    """Dispatch /session subcommands."""
    parts = args.split()
    subcmd = parts[0] if parts else "list"
    rest = parts[1:]

    match subcmd:
        case "list":
            _cmd_list(engine, rest)
        case "info":
            _cmd_info(engine)
        case "delete":
            _cmd_delete(engine, rest)
        case "help":
            engine._out(_HELP, style=STYLE_DIM)
        case _:
            engine._warn(f"Unknown subcommand: {subcmd}")
            engine._out(_HELP, style=STYLE_DIM)


# ── list ─────────────────────────────────────────────────────────────


def _cmd_list(engine: Engine, args: list[str]) -> None:
    """List recent sessions, optionally across all repos."""
    show_all = "--all" in args
    if show_all:
        sessions = engine._store.list_sessions()
    else:
        sessions = engine._store.list_sessions(
            repo_owner=engine.session.owner or None,
            repo_name=engine.session.repo_name or None,
        )

    if not sessions:
        engine._out("No sessions found.")
        return

    current_id = engine.session.session_id
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
    s = engine.session
    repo = f"{s.owner}/{s.repo_name}" if s.owner else "—"
    engine._out(f"  Session ID    {s.session_id[:8]}", style=STYLE_DIM)
    engine._out(f"  Label         {s.session_label or '—'}", style=STYLE_DIM)
    engine._out(f"  Repo          {repo}", style=STYLE_DIM)
    engine._out(f"  Model         {s.model_name or '—'}", style=STYLE_DIM)
    engine._out(f"  Messages      {len(s.message_history)}", style=STYLE_DIM)
    engine._out(f"  Saved         {s.saved_count}", style=STYLE_DIM)


# ── delete ───────────────────────────────────────────────────────────


def _cmd_delete(engine: Engine, args: list[str]) -> None:
    """Delete sessions by ID prefix or age."""
    if not args:
        engine._warn("Usage: /session delete <id> or /session delete --before <duration>")
        return

    if args[0] == "--before":
        _delete_before(engine, args[1:])
    else:
        _delete_by_id(engine, args[0])


def _delete_before(engine: Engine, args: list[str]) -> None:
    """Delete sessions older than a duration."""
    if not args:
        engine._warn("Usage: /session delete --before <duration> (e.g. 7d, 2w, 24h)")
        return

    duration = _parse_duration(args[0])
    if duration is None:
        engine._warn(f"Invalid duration: {args[0]}. Use e.g. 7d, 2w, 24h.")
        return

    cutoff = datetime.now(UTC) - duration
    deleted = engine._store.delete_old_sessions(before=cutoff)
    engine._out(f"Deleted {deleted} message{'s' if deleted != 1 else ''} from old sessions.")


def _delete_by_id(engine: Engine, prefix: str) -> None:
    """Delete a session by ID prefix match."""
    sessions = engine._store.list_sessions(limit=200)
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
    if target.session_id == engine.session.session_id:
        engine._warn("Cannot delete the active session. Use /new first.")
        return

    deleted = engine._store.delete_session(target.session_id)
    label = target.session_label or target.session_id[:8]
    engine._out(f"Deleted session '{label}' ({deleted} messages).")
