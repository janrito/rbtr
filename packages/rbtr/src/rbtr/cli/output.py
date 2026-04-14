"""Output rendering for the rbtr CLI.

All output flows through `emit()`. In JSON mode (piped or
`--json`) models are serialised with `model_dump_json()`. In
TTY mode, rich renders coloured, syntax-highlighted text.

Both modes present the **same data** — the pydantic output model
is the single source of truth. The human format is a richer
layout of the same fields; it never drops or adds information.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from rbtr.cli.models import BuildResult, IndexStatus, SearchHit
from rbtr.config import config
from rbtr.index.models import Chunk, Edge

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)


def lexer_for(file_path: str) -> str:
    """Derive a Pygments lexer name from a file path.

    Delegates to `LanguageManager.get_pygments_lexer`, which
    reads the `pygments_lexer` field from the language
    registration.
    """
    from rbtr.languages import get_manager

    return get_manager().get_pygments_lexer(file_path)


def is_json() -> bool:
    """Whether output should be JSON (piped or ``--json``)."""
    return config.json_output or not sys.stdout.isatty()


def emit(model: BaseModel) -> None:
    """Print a single output model — JSON or rich-formatted."""
    if is_json():
        sys.stdout.write(model.model_dump_json())
        sys.stdout.write("\n")
    else:
        _print_rich(model)


# ── Rich formatting ─────────────────────────────────────────────────


def _score_style(score: float) -> str:
    """Return a rich style string for a search score."""
    if score >= 1.0:
        return "bold green"
    if score >= 0.5:
        return "yellow"
    return "dim"


def _print_rich(model: BaseModel) -> None:
    """Format a model with rich for interactive terminal output.

    Rule: every field in the model must appear in the output.
    """
    match model:
        case BuildResult():
            _render_build_result(model)
        case SearchHit():
            _render_search_hit(model)
        case Chunk():
            _render_chunk(model)
        case Edge():
            _render_edge(model)
        case IndexStatus():
            _render_index_status(model)
        case _:
            console.print(model.model_dump_json())


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
    console.print(t)
    for e in m.errors:
        err_console.print(f"  [red]error:[/] {e}")


def _render_search_hit(m: SearchHit) -> None:
    c = m.chunk
    t = Text()
    t.append(f"  {m.score:.2f}", style=_score_style(m.score))
    t.append(f"  {c.file_path}", style="bold")
    t.append(f":{c.line_start}-{c.line_end}", style="dim")
    t.append(f"  {c.kind}", style="dim")
    t.append(f"  {c.name}")
    t.append(f"  [{c.id[:8]}]", style="dim")
    console.print(t)

    lines = c.content.splitlines()
    preview = "\n".join(lines[:3]) if len(lines) > 3 else c.content
    console.print(
        Syntax(
            preview,
            lexer_for(c.file_path),
            theme="monokai",
            line_numbers=False,
            padding=(0, 2),
        )
    )
    if len(lines) > 3:
        console.print(f"    [dim]... ({len(lines)} lines)[/]")


def _render_chunk(m: Chunk) -> None:
    t = Text()
    t.append(m.file_path, style="bold")
    t.append(f":{m.line_start}-{m.line_end}", style="dim")
    t.append(f"  {m.kind}", style="dim")
    t.append(f"  {m.name}")
    if m.scope:
        t.append(f"  scope={m.scope}", style="dim")
    if m.metadata:
        t.append(f"  {m.metadata}", style="dim")
    console.print(t)
    console.print(
        Syntax(
            m.content,
            lexer_for(m.file_path),
            theme="monokai",
            line_numbers=True,
            start_line=m.line_start,
        )
    )


def _render_edge(m: Edge) -> None:
    t = Text()
    t.append(f"  {m.source_id}")
    t.append(" → ", style="dim")
    t.append(m.target_id)
    t.append(f"  ({m.kind})", style="dim")
    console.print(t)


def _render_index_status(m: IndexStatus) -> None:
    if not m.exists:
        console.print("[red]✗[/] No index found")
    else:
        console.print(f"[green]✓[/] {m.total_chunks} chunks  [dim]{m.db_path}[/]")
