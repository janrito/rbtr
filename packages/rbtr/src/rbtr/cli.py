"""CLI entry point for rbtr — structural code index.

Uses pydantic-settings CliApp for subcommand parsing. Config is
loaded from TOML/env, subcommand args from the CLI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pygit2
from pydantic import BaseModel, Field
from pydantic_settings import CliApp, CliPositionalArg, CliSubCommand, get_subcommand

from rbtr.config import RenderedConfig, config
from rbtr.git import open_repo
from rbtr.workspace import resolve_path

# ── Helpers ──────────────────────────────────────────────────────────


def _open_repo(repo_path: str) -> pygit2.Repository:
    """Open a repo at *repo_path*, or CWD if '.'."""
    if repo_path == ".":
        return open_repo()
    path = pygit2.discover_repository(repo_path)
    if path is None:
        print(f"error: not a git repository: {repo_path}", file=sys.stderr)
        sys.exit(1)
    return pygit2.Repository(path)


def _db_path() -> Path:
    """Resolve the index database path from config."""
    return resolve_path(config.index.db_dir) / "index.duckdb"


def _json(obj: object) -> None:
    """Print a JSON object to stdout."""
    print(json.dumps(obj, default=str))


def _ndjson(items: list[dict[str, object]]) -> None:
    """Print newline-delimited JSON to stdout."""
    for item in items:
        print(json.dumps(item, default=str))


# ── Subcommands ──────────────────────────────────────────────────────


class Build(BaseModel):
    """Build or update the index for a repository."""

    ref: str = Field("HEAD", description="Git ref to index")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.orchestrator import build_index
        from rbtr.index.store import IndexStore

        repo = _open_repo(self.repo_path)
        store = IndexStore(_db_path())

        result = build_index(repo, self.ref, store)
        _json(
            {
                "ref": self.ref,
                "total_files": result.stats.total_files,
                "parsed_files": result.stats.parsed_files,
                "skipped_files": result.stats.skipped_files,
                "total_chunks": result.stats.total_chunks,
                "total_edges": result.stats.total_edges,
                "elapsed_seconds": round(result.stats.elapsed_seconds, 2),
                "errors": result.errors,
            }
        )


class Search(BaseModel):
    """Search the code index."""

    query: CliPositionalArg[str] = Field(description="Search query")
    limit: int = Field(10, description="Maximum results to return")
    ref: str = Field("HEAD", description="Git ref (for diff proximity scoring)")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        store = IndexStore(_db_path())
        results = store.search("HEAD", self.query, top_k=self.limit)
        _ndjson(
            [
                {
                    "id": r.chunk.id,
                    "file_path": r.chunk.file_path,
                    "name": r.chunk.name,
                    "kind": r.chunk.kind.value,
                    "score": round(r.score, 4),
                    "line_start": r.chunk.line_start,
                    "line_end": r.chunk.line_end,
                    "content": r.chunk.content,
                }
                for r in results
            ]
        )


class ReadSymbol(BaseModel):
    """Read a symbol's full source."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name (e.g. HttpClient.retry)")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        store = IndexStore(_db_path())
        chunks = store.search_by_name("HEAD", self.symbol)
        if not chunks:
            print(f"error: symbol not found: {self.symbol}", file=sys.stderr)
            sys.exit(1)
        _ndjson(
            [
                {
                    "file_path": c.file_path,
                    "name": c.name,
                    "kind": c.kind.value,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "content": c.content,
                }
                for c in chunks
            ]
        )


class ListSymbols(BaseModel):
    """List symbols in a file (table of contents)."""

    file: CliPositionalArg[str] = Field(description="File path")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        store = IndexStore(_db_path())
        chunks = store.get_chunks("HEAD", file_path=self.file)
        _ndjson(
            [
                {
                    "name": c.name,
                    "kind": c.kind.value,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                }
                for c in chunks
            ]
        )


class FindRefs(BaseModel):
    """Find references to a symbol via the dependency graph."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        store = IndexStore(_db_path())
        edges = store.get_edges("HEAD", target_id=self.symbol)
        _ndjson(
            [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "kind": e.kind.value,
                }
                for e in edges
            ]
        )


class ChangedSymbols(BaseModel):
    """Show symbols that changed between two refs."""

    base: str = Field(description="Base ref")
    head: str = Field(description="Head ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.git import changed_files
        from rbtr.index.store import IndexStore

        repo = _open_repo(self.repo_path)
        changed = changed_files(repo, self.base, self.head)
        store = IndexStore(_db_path())
        results: list[dict[str, object]] = []
        for path in sorted(changed):
            chunks = store.get_chunks("HEAD", file_path=path)
            for c in chunks:
                results.append(
                    {
                        "file_path": c.file_path,
                        "name": c.name,
                        "kind": c.kind.value,
                        "line_start": c.line_start,
                        "line_end": c.line_end,
                    }
                )
        _ndjson(results)


class Status(BaseModel):
    """Show index status."""

    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.store import IndexStore

        db = _db_path()
        if not db.exists():
            _json({"exists": False})
            return
        store = IndexStore(db)
        total = (
            row[0] if (row := store._cur().execute("SELECT count(*) FROM chunks").fetchone()) else 0
        )
        _json(
            {
                "exists": True,
                "db_path": str(db),
                "total_chunks": total,
            }
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
