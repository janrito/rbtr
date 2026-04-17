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
from rich.progress import Progress
from rich.syntax import Syntax
from rich.text import Text

from rbtr.config import config
from rbtr.daemon.messages import BuildIndexResponse, StatusResponse
from rbtr.index.models import Chunk, Edge
from rbtr.index.search import ScoredResult
from rbtr.languages import get_manager

type ProgressCallback = Callable[[int, int], None]

_out = Console(highlight=False)
_err = Console(stderr=True, highlight=False)


def _json_output() -> bool:
    """Whether stdout should carry JSON (piped or ``--json``)."""
    return config.json_output or not sys.stdout.isatty()


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


@contextmanager
def progress_reporter(
    *,
    parse_label: str = "Parsing files...",
    embed_label: str = "Embedding...",
) -> Iterator[tuple[ProgressCallback | None, ProgressCallback | None]]:
    """Yield ``(on_parse, on_embed)`` progress callbacks.

    When stderr is not a TTY the callbacks are ``None`` — callers
    pass them straight through without needing to know the output
    mode. Otherwise they drive a `rich.progress.Progress` rendered
    on stderr so it never interferes with JSON on stdout.
    """
    if not _err.is_terminal:
        yield None, None
        return

    with Progress(console=_err) as progress:
        parse_task = progress.add_task(f"[cyan]{parse_label}[/]", total=None)
        embed_task = progress.add_task(
            f"[cyan]{embed_label}[/]", total=None, visible=False
        )

        def on_parse(done: int, total: int) -> None:
            progress.update(parse_task, completed=done, total=total)

        def on_embed(done: int, total: int) -> None:
            progress.update(embed_task, completed=done, total=total, visible=True)

        yield on_parse, on_embed


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
        case BuildIndexResponse():
            _render_build_index_response(model)
        case ScoredResult():
            _render_scored_result(model)
        case Chunk():
            _render_chunk(model, compact=compact)
        case Edge():
            _render_edge(model)
        case StatusResponse():
            _render_status_response(model)
        case _:
            msg = f"No rich renderer for {type(model).__name__}"
            raise TypeError(msg)


def _render_build_index_response(m: BuildIndexResponse) -> None:
    s = m.stats
    t = Text()
    t.append("refs=", style="dim")
    t.append(", ".join(m.refs), style="bold")
    t.append(f"  {s.parsed_files}/{s.total_files} files", style="bold")
    t.append(f" ({s.skipped_files} skipped)", style="dim")
    t.append(f"  {s.total_chunks} chunks", style="cyan")
    t.append(f"  {s.total_edges} edges", style="cyan")
    t.append(f"  {s.elapsed_seconds}s", style="dim")
    _out.print(t)
    for e in m.errors:
        print_err(f"  [red]error:[/] {e}")


def _render_scored_result(m: ScoredResult) -> None:
    c = m.chunk
    path = _short_path(c.file_path)
    lexer = get_manager().get_pygments_lexer(c.file_path)

    # Header — same layout as _render_chunk
    t = Text()
    t.append(path, style="bold")
    t.append(f":{c.line_start}-{c.line_end}", style="dim")
    t.append(f"  {c.kind}", style="dim")
    t.append(f"  {c.name}")
    _out.print(t)

    # Score line
    s = Text("  ")
    s.append(f"{m.score:.2f}", style=_score_style(m.score))
    s.append(f"  [{c.id[:8]}]", style="dim")
    _out.print(s)

    # Code preview — skip for single-line chunks (header is enough)
    lines = c.content.splitlines()
    if len(lines) > 1:
        max_preview = 4
        preview = lines[:max_preview]
        if len(lines) > max_preview:
            preview.append(f"… {len(lines)} lines total")
        _out.print(
            Syntax(
                "\n".join(preview),
                lexer,
                theme="monokai",
                line_numbers=False,
                padding=(0, 4),
            )
        )

    _out.print()  # blank line between results


def _render_chunk(m: Chunk, *, compact: bool = False) -> None:
    path = _short_path(m.file_path)
    lexer = get_manager().get_pygments_lexer(m.file_path)

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
            if "module" in meta:
                d.append(meta["module"], style="dim")
            if "names" in meta:
                d.append(f" ({meta['names']})", style="dim")
            if "dots" in meta:
                d.append(f" dots={meta['dots']}", style="dim")
        _out.print(d)

    _out.print(
        Syntax(
            m.content,
            lexer,
            theme="monokai",
            line_numbers=True,
            start_line=m.line_start,
        )
    )
    _out.print()  # blank line between results


def _render_edge(m: Edge) -> None:
    t = Text()
    t.append(f"  {m.source_id}")
    t.append(" → ", style="dim")
    t.append(m.target_id)
    t.append(f"  ({m.kind})", style="dim")
    _out.print(t)


def _render_status_response(m: StatusResponse) -> None:
    if not m.exists:
        _out.print("[red]✗[/] No index found")
    else:
        _out.print(f"[green]✓[/] {m.total_chunks} chunks  [dim]{m.db_path}[/]")
