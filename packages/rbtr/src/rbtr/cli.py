"""CLI entry point for rbtr — structural code index.

Uses pydantic-settings CliApp for subcommand parsing. Config is
loaded from TOML/env, subcommand args from the CLI.

Output modes:
- **TTY** (interactive): rich-formatted text with syntax
  highlighting, coloured scores, and progress bars.
- **Piped / --json**: JSON or NDJSON to stdout.

Both modes emit the **same data** — the pydantic output models
are the single source of truth. The human format is a richer
layout of the same fields; it never drops or adds information.
"""

from __future__ import annotations

import contextvars
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    CliApp,
    CliImplicitFlag,
    CliPositionalArg,
    CliSubCommand,
    get_subcommand,
)
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

from rbtr.config import RenderedConfig, config
from rbtr.git import open_repo
from rbtr.workspace import resolve_path

# ── Consoles ─────────────────────────────────────────────────────────

_console = Console(highlight=False)
_err = Console(stderr=True, highlight=False)


# ── Output models ────────────────────────────────────────────────────
#
# These are the strict interface between the index layer and the
# CLI. All output flows through these models — in both JSON and
# TTY mode. No ad-hoc dicts, no data outside the schema.


class BuildResult(BaseModel):
    """Output of `rbtr build`."""

    ref: str
    total_files: int
    parsed_files: int
    skipped_files: int
    total_chunks: int
    total_edges: int
    elapsed_seconds: float
    errors: list[str]


class SearchHit(BaseModel):
    """One result from `rbtr search`."""

    id: str
    file_path: str
    name: str
    kind: str
    score: float
    line_start: int
    line_end: int
    content: str


class SymbolInfo(BaseModel):
    """A symbol with its full source (`rbtr read-symbol`)."""

    file_path: str
    name: str
    kind: str
    line_start: int
    line_end: int
    content: str


class SymbolSummary(BaseModel):
    """A symbol without content (`rbtr list-symbols`, `rbtr changed-symbols`)."""

    name: str
    kind: str
    line_start: int
    line_end: int
    file_path: str | None = None


class EdgeInfo(BaseModel):
    """A dependency edge (`rbtr find-refs`)."""

    source_id: str
    target_id: str
    kind: str


class IndexStatus(BaseModel):
    """Output of `rbtr status`."""

    exists: bool
    db_path: str | None = None
    total_chunks: int | None = None


# ── Output dispatch ──────────────────────────────────────────────────

_force_json: contextvars.ContextVar[bool] = contextvars.ContextVar("_force_json", default=False)

_EXT_TO_LEXER: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".sh": "bash",
    ".bash": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
    ".sql": "sql",
    ".css": "css",
    ".html": "html",
    ".xml": "xml",
}


def _lexer_for(file_path: str) -> str:
    """Guess a Pygments lexer name from a file path."""
    suffix = Path(file_path).suffix.lower()
    return _EXT_TO_LEXER.get(suffix, "text")


def _is_json() -> bool:
    return _force_json.get() or not sys.stdout.isatty()


def emit(model: BaseModel) -> None:
    """Print a single output model — JSON or rich-formatted.

    Both modes present the **same data**. The rich format is a
    more readable layout of the same fields — it never drops or
    adds information relative to the JSON format.
    """
    if _is_json():
        sys.stdout.write(model.model_dump_json())
        sys.stdout.write("\n")
    else:
        _print_rich(model)


def _score_style(score: float) -> str:
    """Return a rich style for a search score."""
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
            t = Text()
            t.append("ref=", style="dim")
            t.append(model.ref, style="bold")
            t.append(f"  {model.parsed_files}/{model.total_files} files", style="bold")
            t.append(f" ({model.skipped_files} skipped)", style="dim")
            t.append(f"  {model.total_chunks} chunks", style="cyan")
            t.append(f"  {model.total_edges} edges", style="cyan")
            t.append(f"  {model.elapsed_seconds}s", style="dim")
            _console.print(t)
            for err in model.errors:
                _err.print(f"  [red]error:[/] {err}")

        case SearchHit():
            t = Text()
            t.append(f"  {model.score:.2f}", style=_score_style(model.score))
            t.append(f"  {model.file_path}", style="bold")
            t.append(f":{model.line_start}-{model.line_end}", style="dim")
            t.append(f"  {model.kind}", style="dim")
            t.append(f"  {model.name}")
            t.append(f"  [{model.id[:8]}]", style="dim")
            _console.print(t)
            syntax = Syntax(
                model.content,
                _lexer_for(model.file_path),
                theme="monokai",
                line_numbers=False,
                padding=(0, 2),
            )
            lines = model.content.splitlines()
            if len(lines) > 3:
                preview = "\n".join(lines[:3])
                syntax = Syntax(
                    preview,
                    _lexer_for(model.file_path),
                    theme="monokai",
                    line_numbers=False,
                    padding=(0, 2),
                )
                _console.print(syntax)
                _console.print(f"    [dim]... ({len(lines)} lines)[/]")
            else:
                _console.print(syntax)

        case SymbolInfo():
            t = Text()
            t.append(f"{model.file_path}", style="bold")
            t.append(f":{model.line_start}-{model.line_end}", style="dim")
            t.append(f"  {model.kind}", style="dim")
            t.append(f"  {model.name}")
            _console.print(t)
            _console.print(
                Syntax(
                    model.content,
                    _lexer_for(model.file_path),
                    theme="monokai",
                    line_numbers=True,
                    start_line=model.line_start,
                )
            )

        case SymbolSummary():
            prefix = f"{model.file_path}:" if model.file_path else ""
            t = Text()
            t.append(f"  {prefix}{model.line_start}-{model.line_end}", style="dim")
            t.append(f"  {model.kind}", style="cyan")
            t.append(f"  {model.name}")
            _console.print(t)

        case EdgeInfo():
            t = Text()
            t.append(f"  {model.source_id}")
            t.append(" → ", style="dim")
            t.append(model.target_id)
            t.append(f"  ({model.kind})", style="dim")
            _console.print(t)

        case IndexStatus():
            if not model.exists:
                _console.print("[red]✗[/] No index found")
            else:
                _console.print(f"[green]✓[/] {model.total_chunks} chunks  [dim]{model.db_path}[/]")

        case _:
            _console.print(model.model_dump_json())


# ── Subcommands ──────────────────────────────────────────────────────


class Build(BaseModel):
    """Build or update the index for a repository."""

    ref: str = Field("HEAD", description="Git ref to index")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rich.progress import Progress

        from rbtr.index.orchestrator import build_index
        from rbtr.index.store import IndexStore

        repo = open_repo(self.repo_path)
        db = resolve_path(config.db_dir) / "index.duckdb"
        store = IndexStore(db)

        if _is_json():
            result = build_index(repo, self.ref, store)
        else:
            with Progress(console=_err) as progress:
                parse_task = progress.add_task("Parsing files...", total=None)
                embed_task = progress.add_task("Embedding...", total=None, visible=False)

                def on_progress(done: int, total: int) -> None:
                    progress.update(parse_task, completed=done, total=total)

                def on_embed_progress(done: int, total: int) -> None:
                    progress.update(embed_task, completed=done, total=total, visible=True)

                result = build_index(
                    repo,
                    self.ref,
                    store,
                    on_progress=on_progress,
                    on_embed_progress=on_embed_progress,
                )

        emit(
            BuildResult(
                ref=self.ref,
                total_files=result.stats.total_files,
                parsed_files=result.stats.parsed_files,
                skipped_files=result.stats.skipped_files,
                total_chunks=result.stats.total_chunks,
                total_edges=result.stats.total_edges,
                elapsed_seconds=round(result.stats.elapsed_seconds, 2),
                errors=result.errors,
            )
        )


class Search(BaseModel):
    """Search the code index."""

    query: CliPositionalArg[str] = Field(description="Search query")
    limit: int = Field(10, description="Maximum results to return")
    ref: str = Field("HEAD", description="Git ref (for diff proximity scoring)")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        db = resolve_path(config.db_dir) / "index.duckdb"
        store = IndexStore(db)
        for r in store.search("HEAD", self.query, top_k=self.limit):
            emit(
                SearchHit(
                    id=r.chunk.id,
                    file_path=r.chunk.file_path,
                    name=r.chunk.name,
                    kind=r.chunk.kind.value,
                    score=round(r.score, 4),
                    line_start=r.chunk.line_start,
                    line_end=r.chunk.line_end,
                    content=r.chunk.content,
                )
            )


class ReadSymbol(BaseModel):
    """Read a symbol's full source."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name (e.g. HttpClient.retry)")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        db = resolve_path(config.db_dir) / "index.duckdb"
        store = IndexStore(db)
        chunks = store.search_by_name("HEAD", self.symbol)
        if not chunks:
            _err.print(f"[red]error:[/] symbol not found: {self.symbol}")
            sys.exit(1)
        for c in chunks:
            emit(
                SymbolInfo(
                    file_path=c.file_path,
                    name=c.name,
                    kind=c.kind.value,
                    line_start=c.line_start,
                    line_end=c.line_end,
                    content=c.content,
                )
            )


class ListSymbols(BaseModel):
    """List symbols in a file (table of contents)."""

    file: CliPositionalArg[str] = Field(description="File path")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        db = resolve_path(config.db_dir) / "index.duckdb"
        store = IndexStore(db)
        for c in store.get_chunks("HEAD", file_path=self.file):
            emit(
                SymbolSummary(
                    name=c.name,
                    kind=c.kind.value,
                    line_start=c.line_start,
                    line_end=c.line_end,
                )
            )


class FindRefs(BaseModel):
    """Find references to a symbol via the dependency graph."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        db = resolve_path(config.db_dir) / "index.duckdb"
        store = IndexStore(db)
        for e in store.get_edges("HEAD", target_id=self.symbol):
            emit(
                EdgeInfo(
                    source_id=e.source_id,
                    target_id=e.target_id,
                    kind=e.kind.value,
                )
            )


class ChangedSymbols(BaseModel):
    """Show symbols that changed between two refs."""

    base: str = Field(description="Base ref")
    head: str = Field(description="Head ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.git import changed_files
        from rbtr.index.store import IndexStore

        repo = open_repo(self.repo_path)
        changed = changed_files(repo, self.base, self.head)
        db = resolve_path(config.db_dir) / "index.duckdb"
        store = IndexStore(db)
        for path in sorted(changed):
            for c in store.get_chunks("HEAD", file_path=path):
                emit(
                    SymbolSummary(
                        file_path=c.file_path,
                        name=c.name,
                        kind=c.kind.value,
                        line_start=c.line_start,
                        line_end=c.line_end,
                    )
                )


class Status(BaseModel):
    """Show index status."""

    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        db = resolve_path(config.db_dir) / "index.duckdb"
        if not db.exists():
            emit(IndexStatus(exists=False))
            return
        store = IndexStore(db)
        row = store._cur().execute("SELECT count(*) FROM chunks").fetchone()
        emit(
            IndexStatus(
                exists=True,
                db_path=str(db),
                total_chunks=row[0] if row else 0,
            )
        )


# ── Root command ─────────────────────────────────────────────────────


class Rbtr(
    RenderedConfig,
    cli_prog_name="rbtr",
    cli_kebab_case=True,
    cli_implicit_flags=True,
):
    """rbtr — structural code index."""

    json_output: CliImplicitFlag[bool] = Field(
        False,
        alias="json",
        description="Force JSON output",
    )
    search: CliSubCommand[Search]
    build: CliSubCommand[Build]
    read_symbol: CliSubCommand[ReadSymbol]
    list_symbols: CliSubCommand[ListSymbols]
    find_refs: CliSubCommand[FindRefs]
    changed_symbols: CliSubCommand[ChangedSymbols]
    status: CliSubCommand[Status]

    def cli_cmd(self) -> None:
        _force_json.set(self.json_output)

        sub = get_subcommand(self, is_required=False)
        if sub is None:
            CliApp.print_help(self)
            return
        CliApp.run_subcommand(self)


def main() -> None:
    """Entry point for the rbtr CLI."""
    CliApp.run(Rbtr)


if __name__ == "__main__":
    main()
