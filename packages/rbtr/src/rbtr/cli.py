"""CLI entry point for rbtr — structural code index.

Uses pydantic-settings CliApp for subcommand parsing. Config is
loaded from TOML/env, subcommand args from the CLI.

Output modes:
- **TTY** (interactive): human-readable text.
- **Piped / --json**: JSON or NDJSON to stdout.
"""

from __future__ import annotations

import sys

from pydantic import BaseModel, Field
from pydantic_settings import (
    CliApp,
    CliImplicitFlag,
    CliPositionalArg,
    CliSubCommand,
    get_subcommand,
)

from rbtr.config import RenderedConfig, config
from rbtr.git import open_repo
from rbtr.workspace import resolve_path

# ── Output models ────────────────────────────────────────────────────


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

_json_mode = False


def _is_json() -> bool:
    return _json_mode or not sys.stdout.isatty()


def emit(model: BaseModel) -> None:
    """Print a single output model — JSON or human-readable.

    Both modes must present the **same data**. The human format
    is a more readable layout of the same fields — it must never
    drop or add information relative to the JSON format.
    """
    if _is_json():
        print(model.model_dump_json())
    else:
        _print_human(model)


def _print_human(model: BaseModel) -> None:
    """Format a model for interactive terminal output.

    Rule: every field in the model must appear in the output.
    """
    match model:
        case BuildResult():
            print(
                f"ref={model.ref}"
                f"  {model.parsed_files}/{model.total_files} files"
                f" ({model.skipped_files} skipped)"
                f"  {model.total_chunks} chunks"
                f"  {model.total_edges} edges"
                f"  {model.elapsed_seconds}s"
            )
            for err in model.errors:
                print(f"  error: {err}")

        case SearchHit():
            print(
                f"  {model.score:.2f}"
                f"  {model.file_path}:{model.line_start}-{model.line_end}"
                f"  {model.kind}  {model.name}"
                f"  [{model.id}]"
            )
            for line in model.content.splitlines()[:3]:
                print(f"    {line}")
            if model.content.count("\n") > 3:
                print(f"    ... ({model.content.count(chr(10)) + 1} lines)")

        case SymbolInfo():
            print(
                f"{model.file_path}:{model.line_start}-{model.line_end}  {model.kind}  {model.name}"
            )
            print(model.content)

        case SymbolSummary():
            prefix = f"{model.file_path}:" if model.file_path else ""
            print(f"  {prefix}{model.line_start}-{model.line_end}  {model.kind}  {model.name}")

        case EdgeInfo():
            print(f"  {model.source_id} → {model.target_id}  ({model.kind})")

        case IndexStatus():
            if not model.exists:
                print("exists=false")
            else:
                print(f"exists=true  {model.total_chunks} chunks  {model.db_path}")

        case _:
            print(model.model_dump_json())


# ── Subcommands ──────────────────────────────────────────────────────


class Build(BaseModel):
    """Build or update the index for a repository."""

    ref: str = Field("HEAD", description="Git ref to index")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        from rbtr.index.orchestrator import build_index
        from rbtr.index.store import IndexStore

        repo = open_repo(self.repo_path)
        db = resolve_path(config.db_dir) / "index.duckdb"
        store = IndexStore(db)

        result = build_index(repo, self.ref, store)
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
            print(f"error: symbol not found: {self.symbol}", file=sys.stderr)
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
        global _json_mode
        _json_mode = self.json_output

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
