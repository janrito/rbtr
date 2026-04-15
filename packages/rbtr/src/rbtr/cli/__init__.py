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
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    CliApp,
    CliPositionalArg,
    CliSettingsSource,
    CliSubCommand,
    get_subcommand,
)
from rich.progress import Progress
from rich_argparse import RichHelpFormatter

from rbtr.cli.models import BuildResult, IndexStatus
from rbtr.cli.output import _err, emit, is_json, print_err
from rbtr.config import RenderedConfig, config
from rbtr.git import changed_files, open_repo
from rbtr.index.orchestrator import build_index
from rbtr.index.store import IndexStore

# ── Subcommands ──────────────────────────────────────────────────────


class Build(BaseModel):
    """Build or update the index for a repository."""

    ref: str = Field("HEAD", description="Git ref to index")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        repo = open_repo(self.repo_path)
        store = IndexStore.from_config()

        if is_json():
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
                stats=result.stats,
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
        store = IndexStore.from_config()
        for r in store.search("HEAD", self.query, top_k=self.limit):
            emit(r)


class ReadSymbol(BaseModel):
    """Read a symbol's full source."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name (e.g. HttpClient.retry)")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        store = IndexStore.from_config()
        chunks = store.search_by_name("HEAD", self.symbol)
        if not chunks:
            print_err(f"[red]error:[/] symbol not found: {self.symbol}")
            sys.exit(1)
        for c in chunks:
            emit(c)


class ListSymbols(BaseModel):
    """List symbols in a file (table of contents)."""

    file: CliPositionalArg[str] = Field(description="File path")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        store = IndexStore.from_config()
        for c in store.get_chunks("HEAD", file_path=self.file):
            emit(c, compact=True)


class FindRefs(BaseModel):
    """Find references to a symbol via the dependency graph."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        store = IndexStore.from_config()
        for e in store.get_edges("HEAD", target_id=self.symbol):
            emit(e)


class ChangedSymbols(BaseModel):
    """Show symbols that changed between two refs."""

    base: str = Field(description="Base ref")
    head: str = Field(description="Head ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        repo = open_repo(self.repo_path)
        changed = changed_files(repo, self.base, self.head)
        store = IndexStore.from_config()
        for path in sorted(changed):
            for c in store.get_chunks("HEAD", file_path=path):
                emit(c, compact=True)


class Status(BaseModel):
    """Show index status."""

    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        db = Path(config.db_path).expanduser()
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
    cli_source: CliSettingsSource[Rbtr] = CliSettingsSource(Rbtr, formatter_class=RichHelpFormatter)
    CliApp.run(Rbtr, cli_settings_source=cli_source)
