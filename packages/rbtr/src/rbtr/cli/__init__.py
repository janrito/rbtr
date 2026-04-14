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

import sys

from pydantic import BaseModel, Field
from pydantic_settings import (
    CliApp,
    CliPositionalArg,
    CliSubCommand,
    get_subcommand,
)

from rbtr.cli.models import (
    BuildResult,
    EdgeInfo,
    IndexStatus,
    SearchHit,
    SymbolInfo,
    SymbolSummary,
)
from rbtr.cli.output import emit, err_console, is_json
from rbtr.config import RenderedConfig, config
from rbtr.git import open_repo

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
        store = IndexStore.from_config()

        if is_json():
            result = build_index(repo, self.ref, store)
        else:
            with Progress(console=err_console) as progress:
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

        store = IndexStore.from_config()
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

        store = IndexStore.from_config()
        chunks = store.search_by_name("HEAD", self.symbol)
        if not chunks:
            err_console.print(f"[red]error:[/] symbol not found: {self.symbol}")
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

        store = IndexStore.from_config()
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

        store = IndexStore.from_config()
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
        store = IndexStore.from_config()
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
        from rbtr.workspace import resolve_path

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

    search: CliSubCommand[Search]
    build: CliSubCommand[Build]
    read_symbol: CliSubCommand[ReadSymbol]
    list_symbols: CliSubCommand[ListSymbols]
    find_refs: CliSubCommand[FindRefs]
    changed_symbols: CliSubCommand[ChangedSymbols]
    status: CliSubCommand[Status]

    def cli_cmd(self) -> None:
        # Sync parsed CLI values back to the module-level config
        # singleton so output.py and other readers see them.
        config.json_output = self.json_output

        sub = get_subcommand(self, is_required=False)
        if sub is None:
            CliApp.print_help(self)
            return
        CliApp.run_subcommand(self)


def main() -> None:
    """Entry point for the rbtr CLI."""
    CliApp.run(Rbtr)
