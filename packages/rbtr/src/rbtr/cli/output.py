"""Output rendering for the rbtr CLI.

All output flows through `emit()`. In JSON mode (piped or
`--json`) models are serialised with `model_dump_json()`. In
TTY mode, rich renders coloured, syntax-highlighted text.

Both modes present the **same data** — the pydantic output model
is the single source of truth. The human format is a richer
layout of the same fields; it never drops or adds information.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.syntax import Syntax
from rich.text import Text

from rbtr.config import config
from rbtr.daemon.messages import (
    BuildIndexResponse,
    ChangedSymbol,
    ChangedSymbolsResponse,
    GcResponse,
    IndexedRef,
    OkResponse,
    SearchResponse,
    StatusResponse,
)
from rbtr.daemon.status import DaemonStatusReport
from rbtr.index.classify import Expansion
from rbtr.index.models import ChangeKind, Chunk, Edge, ScoredChunk

# Shared change vocabulary with the pi/TUI renderer: sigil + rich
# style per change kind, and the added→modified→removed order.
_CHANGE_DISPLAY: dict[ChangeKind, tuple[str, str]] = {
    ChangeKind.ADDED: ("+", "green"),
    ChangeKind.MODIFIED: ("~", "yellow"),
    ChangeKind.REMOVED: ("\u2212", "red"),
}
_CHANGE_ORDER: dict[ChangeKind, int] = {
    ChangeKind.ADDED: 0,
    ChangeKind.MODIFIED: 1,
    ChangeKind.REMOVED: 2,
}

type ProgressCallback = Callable[[int, int], None]

_out = Console(highlight=False)
_err = Console(stderr=True, highlight=False)


def _json_output() -> bool:
    """Whether stdout should carry JSON (piped or `--json`).

    Follows the stream `_out` actually writes to (matching the
    `_err.is_terminal` check used for progress), so redirecting
    `_out` is enough to select rendered output in tests.
    """
    return config.json_output or not _out.is_terminal


def emit(model: BaseModel, *, compact: bool = False) -> None:
    """Print a single output model — JSON or rich-formatted.

    *compact* only affects TTY mode — JSON is always the full
    model. When True, chunk renderers show a one-line summary
    instead of the full source.
    """
    if _json_output():
        sys.stdout.write(model.model_dump_json())
        sys.stdout.write("\n")
    else:
        _print_rich(model, compact=compact)


def print_err(msg: str) -> None:
    """Print a rich-formatted message to stderr."""
    _err.print(msg)


def print_banner() -> None:
    """Print the rbtr banner to stderr."""
    banner = r"""
            ██        ██
           ░██       ░██
     ██████░██      ██████ ██████
    ░░██░░█░██████ ░░░██░ ░░██░░█
     ░██ ░ ░██░░░██  ░██   ░██ ░
     ░██   ░██  ░██  ░██   ░██
    ░███   ░██████   ░░██ ░███
    ░░░    ░░░░░      ░░  ░░░
"""
    _err.print(banner, end="\n")


def _noop_progress(_done: int, _total: int) -> None:
    """No-op callback used when stderr isn't a TTY."""


def _make_progress_callback(progress: Progress, task_id: TaskID) -> ProgressCallback:
    def on_update(done: int, total: int) -> None:
        progress.update(task_id, completed=done, total=total, visible=True)

    return on_update


@contextmanager
def progress_reporter(*labels: str) -> Iterator[list[ProgressCallback]]:
    """Yield one progress callback per task *label*, in order.

    When stderr is not a TTY every yielded callback is a no-op,
    so callers can invoke them unconditionally. Otherwise the
    callbacks drive a `rich.progress.Progress` on stderr so it
    never interferes with JSON on stdout.

    Tasks start hidden and become visible on their first update.
    """
    if not _err.is_terminal:
        yield [_noop_progress for _ in labels]
        return

    with Progress(console=_err) as progress:
        task_ids = [
            progress.add_task(f"[cyan]{label}[/]", total=None, visible=False) for label in labels
        ]
        yield [_make_progress_callback(progress, tid) for tid in task_ids]


# ── Helpers ──────────────────────────────────────────────────────────


def _short_path(file_path: str) -> str:
    """Shorten a file path relative to CWD when possible."""
    try:
        return os.path.relpath(file_path)
    except ValueError:
        return file_path


def _score_style(score: float) -> str:
    """Return a rich style string for a search score."""
    if score >= 1.0:
        return "bold green"
    if score >= 0.5:
        return "yellow"
    return "dim"


# ── Rich formatting ─────────────────────────────────────────────────


def _print_rich(model: BaseModel, *, compact: bool = False) -> None:
    """Format a model with rich for interactive terminal output.

    Rule: every field in the model must appear in the output.
    """
    match model:
        case OkResponse():
            _out.print("[green]ok[/]")
        case BuildIndexResponse():
            _render_build_index_response(model)
        case SearchResponse():
            _render_search_response(model)
        case ScoredChunk():
            _render_scored_result(model)
        case Chunk():
            _render_chunk(model, compact=compact)
        case ChangedSymbolsResponse():
            _render_changed_symbols_response(model)
        case Edge():
            _render_edge(model)
        case StatusResponse():
            _render_status_response(model)
        case DaemonStatusReport():
            _render_daemon_status_report(model)
        case GcResponse():
            _render_gc_response(model)
        case _:
            msg = f"No rich renderer for {type(model).__name__}"
            raise TypeError(msg)


def _render_build_index_response(m: BuildIndexResponse) -> None:
    s = m.stats
    t = Text()
    t.append("refs=", style="dim")
    t.append(", ".join(m.resolved_refs), style="bold")
    t.append(f"  {s.parsed_files}/{s.total_files} files", style="bold")
    t.append(f" ({s.skipped_files} skipped)", style="dim")
    t.append(f"  {s.total_chunks} chunks", style="cyan")
    t.append(f"  {s.total_edges} edges", style="cyan")
    t.append(f"  {s.elapsed_seconds}s", style="dim")
    _out.print(t)
    for e in m.errors:
        print_err(f"  [red]error:[/] {e}")


def _render_search_response(m: SearchResponse) -> None:
    if m.expansion is not None:
        _render_expansion_summary(m.expansion)
    for r in m.results:
        _render_scored_result(r)


def _render_expansion_summary(exp: Expansion) -> None:
    """Print a dim one-liner summarising the expansion applied to a search."""
    t = Text("  expansion: ", style="dim")
    t.append(exp.kind.value, style="dim bold")
    if exp.keywords:
        t.append(f"  kw=[{', '.join(exp.keywords)}]", style="dim")
    if exp.variants:
        t.append(f"  variants={len(exp.variants)}", style="dim")
    _out.print(t)


def _render_scored_result(m: ScoredChunk) -> None:
    path = _short_path(m.file_path)
    if m.repo_path is not None:
        path = f"{os.path.basename(m.repo_path.rstrip('/'))}/{path}"

    # Header — same layout as _render_chunk
    t = Text()
    t.append(path, style="bold")
    t.append(f":{m.line_start}-{m.line_end}", style="dim")
    t.append(f"  {m.kind}", style="dim")
    t.append(f"  {m.name}")
    _out.print(t)

    # Score line
    s = Text("  ")
    s.append(f"{m.score:.2f}", style=_score_style(m.score))
    s.append(f"  [{m.id[:8]}]", style="dim")
    _out.print(s)

    # Code preview — skip for single-line chunks (header is enough)
    lines = m.content.splitlines()
    if len(lines) > 1:
        max_preview = 4
        preview = lines[:max_preview]
        if len(lines) > max_preview:
            preview.append(f"… {len(lines)} lines total")
        _out.print(
            Syntax(
                "\n".join(preview),
                m.language,
                theme="monokai",
                line_numbers=False,
                padding=(0, 4),
            )
        )

    _out.print()  # blank line between results


def _render_chunk(m: Chunk, *, compact: bool = False) -> None:
    path = _short_path(m.file_path)

    if compact:
        # One-line summary for list-symbols / changed-symbols
        t = Text()
        t.append(f"  {m.line_start:>4}-{m.line_end:<4}", style="dim")
        t.append(f"  {m.kind:<10}", style="cyan")
        t.append(m.name)
        if m.scope:
            t.append(f"  ({m.scope})", style="dim")
        _out.print(t)
        return

    # Full view for read-symbol — same header structure as search
    t = Text()
    t.append(path, style="bold")
    t.append(f":{m.line_start}-{m.line_end}", style="dim")
    t.append(f"  {m.kind}", style="dim")
    t.append(f"  {m.name}")
    _out.print(t)

    # Detail line — scope and metadata
    if m.scope or m.metadata:
        d = Text("  ")
        if m.scope:
            d.append(m.scope, style="dim")
        if m.metadata:
            meta = m.metadata
            if m.scope:
                d.append("  ", style="dim")
            if meta.module:
                d.append(meta.module, style="dim")
            if meta.names:
                d.append(f" ({meta.names})", style="dim")
            if meta.dots:
                d.append(f" dots={meta.dots}", style="dim")
        _out.print(d)

    _out.print(
        Syntax(
            m.content,
            m.language,
            theme="monokai",
            line_numbers=True,
            start_line=m.line_start,
        )
    )
    _out.print()  # blank line between results


def _render_changed_symbol(item: ChangedSymbol) -> None:
    """One labelled symbol row: coloured sigil + kind + name + scope + path.

    No line range — for a removed symbol it would be base-side and
    misleading, so changed-symbols rows omit it (unlike read/list).
    """
    sigil, style = _CHANGE_DISPLAY[item.change]
    c = item.chunk
    t = Text()
    t.append(f"{sigil} ", style=style)
    t.append(f"{c.kind:<10}", style="cyan")
    t.append(c.name)
    if c.scope:
        t.append(f"  ({c.scope})", style="dim")
    t.append(f"  {_short_path(c.file_path)}", style="dim")
    _out.print(t)


def _render_changed_symbols_response(m: ChangedSymbolsResponse) -> None:
    if not m.changes:
        _out.print("[dim]No changed symbols.[/]")
        return
    counts: dict[ChangeKind, int] = dict.fromkeys(ChangeKind, 0)
    for item in sorted(m.changes, key=lambda i: _CHANGE_ORDER[i.change]):
        _render_changed_symbol(item)
        counts[item.change] += 1
    summary = Text("\n")
    summary.append(f"+{counts[ChangeKind.ADDED]}", style="green")
    summary.append(f"  ~{counts[ChangeKind.MODIFIED]}", style="yellow")
    summary.append(f"  \u2212{counts[ChangeKind.REMOVED]}", style="red")
    _out.print(summary)


def _render_edge(m: Edge) -> None:
    t = Text()
    t.append(f"  {m.source_id}")
    t.append(" → ", style="dim")
    t.append(m.target_id)
    t.append(f"  ({m.kind})", style="dim")
    _out.print(t)


def _render_daemon_status_report(m: DaemonStatusReport) -> None:
    if not m.running:
        _out.print("[red]✗[/]  Daemon not running")
        return
    t = Text.from_markup("[green]✓[/]  Daemon running")
    t.append(f"  pid={m.pid}", style="dim")
    if m.version is not None:
        t.append(f"  v{m.version}", style="dim")
    if m.uptime_seconds is not None:
        t.append(f"  up {m.uptime_seconds:.1f}s", style="dim")
    _out.print(t)
    if m.rpc is not None:
        _out.print(f"   [dim]rpc:[/] {m.rpc}")
    if m.pub is not None:
        _out.print(f"   [dim]pub:[/] {m.pub}")


def _render_status_response(m: StatusResponse) -> None:
    # Output is derived solely from the response model.
    print_banner()
    if not m.indexed_refs:
        _out.print("[red]✗[/]  No index found")
    elif len({ref.repo_path for ref in m.indexed_refs}) > 1:
        # Cross-repo: group refs under their repo.
        _out.print(f"[green]✓[/]  indexed repos  [dim]{m.db_path}[/]")
        by_repo: dict[str, list[IndexedRef]] = {}
        for ref in m.indexed_refs:
            by_repo.setdefault(ref.repo_path or "?", []).append(ref)
        for repo_path, refs in by_repo.items():
            _out.print(f"  [bold]{repo_path}[/]")
            for ref in refs:
                _out.print(f"     {_fmt_ref(ref)}")
    else:
        total = m.indexed_refs[0].total
        _out.print(f"[green]✓[/]  {_human(total)} chunks  [dim]{m.db_path}[/]")
        for ref in m.indexed_refs:
            _out.print(f"   {_fmt_ref(ref)}")
    if m.active_build is not None:
        job = m.active_build
        pct = f" ({100 * job.current / job.total:.0f}%)" if job.total > 0 else ""
        elapsed = _format_elapsed(job.elapsed_seconds)
        _out.print(
            f"[cyan]⟳[/]  Building: {job.ref[:12]} — "
            f"{job.phase} {job.current}/{job.total}{pct} — {elapsed}"
        )
    if m.active_embed is not None:
        ej = m.active_embed
        pct = f" ({100 * ej.current / ej.total:.0f}%)" if ej.total > 0 else ""
        elapsed = _format_elapsed(ej.elapsed_seconds)
        _out.print(
            f"[magenta]↻[/]  Embedding: {ej.ref[:12]} — {ej.current}/{ej.total}{pct} — {elapsed}"
        )


def _fmt_ref(ref: IndexedRef) -> str:
    """Render one indexed ref as a single line."""
    short = ref.sha[:12]
    label = f"{short} ({', '.join(ref.names)})" if ref.names else short
    parts = [label, f"{_human(ref.total)} indexed"]
    if ref.embedded >= ref.total:
        parts.append(f"{_human(ref.embedded)} embedded [green]✓[/]")
    elif ref.embedded > 0:
        pct = 100 * ref.embedded / ref.total
        parts.append(f"{_human(ref.embedded)} embedded [yellow]{pct:.0f}%[/]")
    else:
        parts.append("[yellow]not embedded[/]")
    return "  ".join(parts)


def _human(n: int) -> str:
    """Format a count for humans: `42`, `1.2k`, `11.2k`."""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


def _format_elapsed(seconds: float) -> str:
    """Format a monotonic-seconds delta as `1m23s` / `45s`."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def _render_gc_response(m: GcResponse) -> None:
    prefix = "[yellow]would remove[/]" if m.dry_run else "[green]removed[/]"
    t = Text.from_markup(
        f"{prefix}  {m.commits_dropped} commits, "
        f"{m.snapshots_dropped} snapshots, "
        f"{m.edges_dropped} edges, "
        f"{m.chunks_dropped} chunks"
    )
    t.append(f"  ({m.elapsed_seconds:.2f}s)", style="dim")
    _out.print(t)
