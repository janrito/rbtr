"""Handler for /stats — session token, cost, and tool statistics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rbtr_legacy.config import config
from rbtr_legacy.sessions.kinds import GLOBAL_SCOPE
from rbtr_legacy.sessions.stats import IncidentStats, OverheadStats, TokenStats, ToolStat
from rbtr_legacy.usage import format_cost, format_tokens

if TYPE_CHECKING:
    from .core import Engine

_COL = 10  # column width for numeric values


def cmd_stats(engine: Engine, args: str) -> None:
    """Dispatch /stats subcommands."""
    arg = args.strip()
    if arg in ("all", "--all"):
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
    incidents = engine.store.incident_stats(sid)
    oh = engine.store.overhead_stats(sid)

    _out(engine, f"Session ({_elapsed(engine)})")
    _out(engine, _row("Model", engine.state.model_name or "—"))
    _render_body(engine, ts, tools, incidents, oh, show_context=True)
    _render_facts(engine)
    engine._context(
        f"[/stats → {ts.total_turns} turns]",
        f"Viewed session stats: {ts.total_turns} turns, {ts.total_responses} responses.",
    )


# ── /stats <session_id> ─────────────────────────────────────────────


def _cmd_historical(engine: Engine, query: str) -> None:
    from rbtr_legacy.engine.session_cmd import _find_session

    target = _find_session(engine, query)
    if target is None:
        return
    sid = target.session_id
    ts = engine.store.token_stats(sid)
    tools = engine.store.tool_stats(sid)
    incidents = engine.store.incident_stats(sid)
    oh = engine.store.overhead_stats(sid)

    _out(engine, f"Session {target.session_label or sid[:8]}")
    _out(engine, _row("ID", sid[:8]))
    if target.model_name:
        _out(engine, _row("Model", target.model_name))
    _render_body(engine, ts, tools, incidents, oh)


# ── /stats all ───────────────────────────────────────────────────────


def _cmd_global(engine: Engine) -> None:
    gs = engine.store.global_stats()
    if gs.session_count == 0:
        engine._out("No sessions found.")
        return

    incidents = engine.store.global_incident_stats()
    oh = engine.store.global_overhead_stats()

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

    _render_tools(engine, gs.tools, compact=False)

    if oh.has_overhead:
        _render_overhead(engine, oh)

    if incidents.has_incidents:
        _render_incidents(engine, incidents)

    _render_facts(engine, all_scopes=True)


# ── Shared rendering ────────────────────────────────────────────────


def _out(engine: Engine, text: str) -> None:
    engine._out(text)


def _elapsed(engine: Engine) -> str:
    secs = int(time.time() - engine.state.session_started_at)
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
    incidents: IncidentStats,
    oh: OverheadStats,
    *,
    show_context: bool = False,
) -> None:
    compact = ts.compaction_count > 0

    if compact:
        _out(engine, _row("Compactions", str(ts.compaction_count)))

    _render_messages(engine, ts, compact)
    _render_tokens(engine, ts, compact, show_context=show_context)
    _render_cost(engine, ts, compact)
    _render_tools(engine, tools, compact)
    if oh.has_overhead:
        _render_overhead(engine, oh)
    if incidents.has_incidents:
        _render_incidents(engine, incidents)


def _render_messages(engine: Engine, ts: TokenStats, compact: bool) -> None:
    def act(val: int) -> str:
        return str(val) if compact else ""

    _out(engine, "")
    if compact:
        _out(engine, _row("", "total", "active"))
    _out(engine, _row("Turns", str(ts.total_turns), act(ts.active_turns)))
    _out(engine, _row("Responses", str(ts.total_responses), act(ts.active_responses)))


def _render_tokens(
    engine: Engine, ts: TokenStats, compact: bool, *, show_context: bool = False
) -> None:
    if not ts.total_input_tokens and not ts.total_output_tokens:
        return

    def act(val: int) -> str:
        return format_tokens(val) if compact else ""

    _out(engine, "")
    if compact:
        _out(engine, _row("Tokens", "total", "active"))
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


def _render_cost(engine: Engine, ts: TokenStats, compact: bool) -> None:
    if not ts.total_cost:
        return

    _out(engine, "")
    if compact:
        _out(engine, _row("Cost", "total", "active"))
    _out(
        engine,
        _row("Total", format_cost(ts.total_cost), format_cost(ts.active_cost) if compact else ""),
    )


def _render_overhead(engine: Engine, oh: OverheadStats) -> None:
    if oh.compaction_count:
        _out(engine, "")
        _out(engine, f"  Compaction overhead ({oh.compaction_count})")
        _out(engine, _row("Input", format_tokens(oh.compaction_input_tokens)))
        _out(engine, _row("Output", format_tokens(oh.compaction_output_tokens)))
        if oh.compaction_cost:
            _out(engine, _row("Cost", format_cost(oh.compaction_cost)))

    if oh.fact_extraction_count:
        _out(engine, "")
        _out(engine, f"  Fact extraction overhead ({oh.fact_extraction_count})")
        _out(engine, _row("Input", format_tokens(oh.fact_extraction_input_tokens)))
        _out(engine, _row("Output", format_tokens(oh.fact_extraction_output_tokens)))
        if oh.fact_extraction_cost:
            _out(engine, _row("Cost", format_cost(oh.fact_extraction_cost)))


def _render_facts(engine: Engine, *, all_scopes: bool = False) -> None:
    """Render fact counts.

    *all_scopes*: show every scope (`/stats all`).
    Otherwise show only global + current repo (`/stats`).
    """
    if not config.memory.enabled:
        return
    counts = engine.store.fact_counts()
    if not counts:
        return

    if all_scopes:
        visible = counts
    else:
        scopes = {GLOBAL_SCOPE}
        repo_scope = engine.state.repo_scope
        if repo_scope:
            scopes.add(repo_scope)
        visible = {s: c for s, c in counts.items() if s in scopes}

    if not visible:
        return
    total = sum(visible.values())
    _out(engine, "")
    _out(engine, f"Facts ({total})")
    for scope, count in sorted(visible.items()):
        label = "global" if scope == GLOBAL_SCOPE else scope
        _out(engine, _row(label, str(count)))


def _render_tools(engine: Engine, tools: list[ToolStat], compact: bool) -> None:
    if not tools:
        return
    total_calls = sum(t.call_count for t in tools)
    active_calls = sum(t.active_call_count for t in tools)

    _out(engine, "")
    if compact:
        _out(engine, _row("Tools", "total", "active"))
        _out(engine, _row("Calls", str(total_calls), str(active_calls)))
    else:
        _out(engine, f"  Tool calls ({total_calls})")

    for t in tools:
        act = str(t.active_call_count) if compact else ""
        fail = f"  ({t.failure_count} failed)" if t.failure_count else ""
        _out(engine, _row(f"  {t.tool_name}", str(t.call_count), act) + fail)


def _render_incidents(engine: Engine, incidents: IncidentStats) -> None:
    if incidents.failures:
        total = incidents.total_failures
        _out(engine, "")
        _out(engine, f"  Failures ({total})")
        for f in incidents.failures:
            parts: list[str] = []
            if f.recovered_count:
                parts.append(f"recovered: {f.recovered_count}")
            if f.failed_count:
                parts.append(f"failed: {f.failed_count}")
            suffix = f"   {', '.join(parts)}" if parts else ""
            _out(engine, f"    {f.failure_kind:<24}{f.total_count:>{_COL}}{suffix}")

    if incidents.repairs:
        total = sum(r.total_count for r in incidents.repairs)
        _out(engine, "")
        _out(engine, f"  History repairs ({total})")
        for r in incidents.repairs:
            reason = f"   ({r.reason})" if r.reason else ""
            _out(engine, f"    {r.strategy:<24}{r.total_count:>{_COL}}{reason}")

    if incidents.failures:
        total = incidents.total_failures
        recovered = incidents.total_recovered
        if total > 0:
            pct = recovered / total * 100
            _out(engine, "")
            _out(engine, f"  Recovery rate       {pct:5.0f}%   {recovered}/{total}")
