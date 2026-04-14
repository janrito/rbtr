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

from rbtr.cli.models import (
    BuildResult,
    EdgeInfo,
    IndexStatus,
    SearchHit,
    SymbolInfo,
    SymbolSummary,
)
from rbtr.config import config

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

# Language IDs that don't match their Pygments lexer name.
_LEXER_OVERRIDES: dict[str, str] = {
    "c_sharp": "csharp",
}


def lexer_for(file_path: str) -> str:
    """Derive a Pygments lexer name from a file path.

    Uses the language manager's extension map — no duplicate
    mapping to maintain.
    """
    from rbtr.languages import get_manager

    lang_id = get_manager().detect_language(file_path)
    if lang_id is None:
        return "text"
    return _LEXER_OVERRIDES.get(lang_id, lang_id)


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
        case SymbolInfo():
            _render_symbol_info(model)
        case SymbolSummary():
            _render_symbol_summary(model)
        case EdgeInfo():
            _render_edge_info(model)
        case IndexStatus():
            _render_index_status(model)
        case _:
            console.print(model.model_dump_json())


def _render_build_result(m: BuildResult) -> None:
    t = Text()
    t.append("ref=", style="dim")
    t.append(m.ref, style="bold")
    t.append(f"  {m.parsed_files}/{m.total_files} files", style="bold")
    t.append(f" ({m.skipped_files} skipped)", style="dim")
    t.append(f"  {m.total_chunks} chunks", style="cyan")
    t.append(f"  {m.total_edges} edges", style="cyan")
    t.append(f"  {m.elapsed_seconds}s", style="dim")
    console.print(t)
    for e in m.errors:
        err_console.print(f"  [red]error:[/] {e}")


def _render_search_hit(m: SearchHit) -> None:
    t = Text()
    t.append(f"  {m.score:.2f}", style=_score_style(m.score))
    t.append(f"  {m.file_path}", style="bold")
    t.append(f":{m.line_start}-{m.line_end}", style="dim")
    t.append(f"  {m.kind}", style="dim")
    t.append(f"  {m.name}")
    t.append(f"  [{m.id[:8]}]", style="dim")
    console.print(t)

    lines = m.content.splitlines()
    preview = "\n".join(lines[:3]) if len(lines) > 3 else m.content
    console.print(
        Syntax(
            preview,
            lexer_for(m.file_path),
            theme="monokai",
            line_numbers=False,
            padding=(0, 2),
        )
    )
    if len(lines) > 3:
        console.print(f"    [dim]... ({len(lines)} lines)[/]")


def _render_symbol_info(m: SymbolInfo) -> None:
    t = Text()
    t.append(m.file_path, style="bold")
    t.append(f":{m.line_start}-{m.line_end}", style="dim")
    t.append(f"  {m.kind}", style="dim")
    t.append(f"  {m.name}")
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


def _render_symbol_summary(m: SymbolSummary) -> None:
    prefix = f"{m.file_path}:" if m.file_path else ""
    t = Text()
    t.append(f"  {prefix}{m.line_start}-{m.line_end}", style="dim")
    t.append(f"  {m.kind}", style="cyan")
    t.append(f"  {m.name}")
    console.print(t)


def _render_edge_info(m: EdgeInfo) -> None:
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
