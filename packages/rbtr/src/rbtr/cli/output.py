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

from pydantic import BaseModel
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from rbtr.cli.models import BuildResult, IndexStatus
from rbtr.config import config
from rbtr.index.models import Chunk, Edge
from rbtr.index.search import ScoredResult
from rbtr.languages import get_manager

_out = Console(highlight=False)
_err = Console(stderr=True, highlight=False)


def is_json() -> bool:
    """Whether output should be JSON (piped or ``--json``)."""
    return config.json_output or not sys.stdout.isatty()


def emit(model: BaseModel, *, compact: bool = False) -> None:
    """Print a single output model — JSON or rich-formatted.

    *compact* only affects TTY mode — JSON is always the full
    model. When True, chunk renderers show a one-line summary
    instead of the full source.
    """
    if is_json():
        sys.stdout.write(model.model_dump_json())
        sys.stdout.write("\n")
    else:
        _print_rich(model, compact=compact)


def print_err(msg: str) -> None:
    """Print a rich-formatted message to stderr."""
    _err.print(msg)


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
        case BuildResult():
            _render_build_result(model)
        case ScoredResult():
            _render_scored_result(model)
        case Chunk():
            _render_chunk(model, compact=compact)
        case Edge():
            _render_edge(model)
        case IndexStatus():
            _render_index_status(model)
        case _:
            _out.print(model.model_dump_json())


def _render_build_result(m: BuildResult) -> None:
    s = m.stats
    t = Text()
    t.append("ref=", style="dim")
    t.append(m.ref, style="bold")
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

    # Header line
    t = Text()
    t.append(f"{m.score:.2f}", style=_score_style(m.score))
    t.append(f"  {path}", style="bold")
    t.append(f":{c.line_start}-{c.line_end}", style="dim")
    t.append(f"  {c.kind}", style="dim")
    t.append(f"  {c.name}")
    t.append(f"  [{c.id[:8]}]", style="dim")
    _out.print(t)

    # Code preview — skip for single-line chunks (header is enough)
    lines = c.content.splitlines()
    if len(lines) > 1:
        preview_lines = min(len(lines), 4)
        preview = "\n".join(lines[:preview_lines])
        _out.print(
            Syntax(
                preview,
                lexer,
                theme="monokai",
                line_numbers=False,
                padding=(0, 4),
            )
        )
        if len(lines) > preview_lines:
            _out.print(f"      [dim]… {len(lines)} lines total[/]")

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

    # Full view for read-symbol
    t = Text()
    t.append(path, style="bold")
    t.append(f":{m.line_start}-{m.line_end}", style="dim")
    t.append(f"  {m.kind}", style="dim")
    t.append(f"  {m.name}")
    if m.scope:
        t.append(f"  ({m.scope})", style="dim")
    _out.print(t)

    if m.metadata:
        _out.print(Text(f"  {m.metadata}", style="dim"))

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


def _render_index_status(m: IndexStatus) -> None:
    if not m.exists:
        _out.print("[red]✗[/] No index found")
    else:
        _out.print(f"[green]✓[/] {m.total_chunks} chunks  [dim]{m.db_path}[/]")
