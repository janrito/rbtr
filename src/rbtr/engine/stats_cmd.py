"""Handler for /stats — session token, cost, and tool statistics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rbtr.sessions.stats import TokenStats, ToolStat
from rbtr.styles import STYLE_DIM
from rbtr.usage import format_cost, format_tokens

if TYPE_CHECKING:
    from .core import Engine

_COL = 10  # column width for numeric values


def cmd_stats(engine: Engine, args: str) -> None:
    """Dispatch /stats subcommands."""
    arg = args.strip()
    if arg == "--all":
        _cmd_global(engine)
    elif arg:
        _cmd_historical(engine, arg)
    else:
        _cmd_current(engine)


# ── /stats (current session) ────────────────────────────────────────


def _cmd_current(engine: Engine) -> None:
    sid = engine.state.session_id
    ts = engine.store.token_stats(sid)
    tools = engine.store.tool_stats(sid)

    _out(engine, f"Session ({_elapsed(engine)})")
    _out(engine, _row("Model", engine.state.model_name or "—"))
    _render_body(engine, ts, tools, show_context=True)


# ── /stats <session_id> ─────────────────────────────────────────────


def _cmd_historical(engine: Engine, prefix: str) -> None:
    sessions = engine.store.list_sessions(limit=200)
    matches = [s for s in sessions if s.session_id.startswith(prefix)]

    if not matches:
        engine._warn(f"No session matching '{prefix}'.")
        return
    if len(matches) > 1:
        engine._warn(f"Ambiguous prefix '{prefix}' — matches {len(matches)} sessions.")
        for s in matches[:5]:
            _out(engine, f"  {s.session_id[:12]}  {s.session_label or '—'}")
        return

    target = matches[0]
    sid = target.session_id
    ts = engine.store.token_stats(sid)
    tools = engine.store.tool_stats(sid)

    _out(engine, f"Session {target.session_label or sid[:8]}")
    _out(engine, _row("ID", sid[:8]))
    if target.model_name:
        _out(engine, _row("Model", target.model_name))
    _render_body(engine, ts, tools)


# ── /stats --all ─────────────────────────────────────────────────────


def _cmd_global(engine: Engine) -> None:
    gs = engine.store.global_stats()
    if gs.session_count == 0:
        engine._out("No sessions found.")
        return

    _out(engine, f"All sessions ({gs.session_count})")
    _out(engine, _row("Total cost", format_cost(gs.total_cost) if gs.total_cost else "—"))
    _out(engine, _row("Input tokens", format_tokens(gs.total_input_tokens)))
    _out(engine, _row("Output tokens", format_tokens(gs.total_output_tokens)))

    if gs.total_cache_read_tokens:
        hit = _hit_rate(gs.total_cache_read_tokens, gs.total_input_tokens)
        _out(engine, _row("Cache read", format_tokens(gs.total_cache_read_tokens), suffix=hit))
    if gs.total_cache_write_tokens:
        _out(engine, _row("Cache write", format_tokens(gs.total_cache_write_tokens)))

    if gs.models:
        _out(engine, "")
        _out(engine, "  By model")
        for m in gs.models:
            cost = format_cost(m.total_cost) if m.total_cost else "—"
            _out(engine, f"    {m.model_name or '?':<30}{cost:>10}   ({m.session_count} sessions)")

    _render_tools(engine, gs.tools)


# ── Shared rendering ────────────────────────────────────────────────


def _out(engine: Engine, text: str) -> None:
    engine._out(text, style=STYLE_DIM)


def _elapsed(engine: Engine) -> str:
    secs = int(time.monotonic() - engine.state.session_started_at)
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins:02d}m"


def _row(label: str, total: str, active: str = "", suffix: str = "") -> str:
    base = f"  {label:<16}{total:>{_COL}}"
    if active:
        base += f"  {active:>{_COL}}"
    if suffix:
        base += f"   {suffix}"
    return base


def _hit_rate(cache_read: int, input_total: int) -> str:
    if not input_total:
        return ""
    return f"{cache_read / input_total * 100:.0f}% hit rate"


def _render_body(
    engine: Engine,
    ts: TokenStats,
    tools: list[ToolStat],
    *,
    show_context: bool = False,
) -> None:
    _render_messages(engine, ts)
    _render_tokens(engine, ts, show_context=show_context)
    _render_cost(engine, ts)
    _render_tools(engine, tools)


def _render_messages(engine: Engine, ts: TokenStats) -> None:
    if ts.compaction_count > 0:
        compacted = ts.total_messages - ts.active_messages
        _out(
            engine,
            _row(
                "Messages",
                str(ts.total_messages),
                suffix=f"({ts.active_messages} active, {compacted} compacted)",
            ),
        )
        _out(engine, _row("Compactions", str(ts.compaction_count)))
    else:
        _out(engine, _row("Messages", str(ts.total_messages)))


def _render_tokens(engine: Engine, ts: TokenStats, *, show_context: bool = False) -> None:
    if not ts.total_input_tokens and not ts.total_output_tokens:
        return

    compact = ts.compaction_count > 0

    def act(val: int) -> str:
        return format_tokens(val) if compact else ""

    _out(engine, "")
    _out(engine, _row("Tokens", "total" if compact else "", "active" if compact else ""))
    _out(engine, _row("Input", format_tokens(ts.total_input_tokens), act(ts.active_input_tokens)))
    _out(
        engine, _row("Output", format_tokens(ts.total_output_tokens), act(ts.active_output_tokens))
    )

    if ts.total_cache_read_tokens:
        hit = _hit_rate(ts.total_cache_read_tokens, ts.total_input_tokens)
        _out(
            engine,
            _row(
                "Cache read",
                format_tokens(ts.total_cache_read_tokens),
                act(ts.active_cache_read_tokens),
                suffix=hit,
            ),
        )
    if ts.total_cache_write_tokens:
        _out(
            engine,
            _row(
                "Cache write",
                format_tokens(ts.total_cache_write_tokens),
                act(ts.active_cache_write_tokens),
            ),
        )

    if show_context:
        usage = engine.state.usage
        if usage.context_window_known:
            _out(
                engine,
                _row(
                    "Context",
                    f"{usage.context_used_pct:.0f}% of {format_tokens(usage.context_window)}",
                ),
            )


def _render_cost(engine: Engine, ts: TokenStats) -> None:
    if not ts.total_cost:
        return

    compact = ts.compaction_count > 0
    _out(engine, "")
    _out(engine, _row("Cost", "total" if compact else "", "active" if compact else ""))
    _out(
        engine,
        _row("Total", format_cost(ts.total_cost), format_cost(ts.active_cost) if compact else ""),
    )


def _render_tools(engine: Engine, tools: list[ToolStat]) -> None:
    if not tools:
        return
    total_calls = sum(t.call_count for t in tools)
    _out(engine, "")
    _out(engine, f"  Tool calls ({total_calls})")
    for t in tools:
        fail = f"  ({t.failure_count} failed)" if t.failure_count else ""
        _out(engine, f"    {t.tool_name:<20}{t.call_count:>4}{fail}")
