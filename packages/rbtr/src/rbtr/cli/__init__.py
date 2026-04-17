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

Dual-mode: most commands try the daemon first; if unreachable,
fall back to direct in-process execution. `rbtr index` auto-starts
the daemon unless `--no-daemon` is given.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import json

from pydantic import BaseModel, Field, TypeAdapter
from pydantic_settings import (
    CliApp,
    CliPositionalArg,
    CliSettingsSource,
    CliSubCommand,
    get_subcommand,
)
from rich_argparse import RichHelpFormatter

from rbtr.cli.output import emit, print_err, progress_reporter
from rbtr.config import Config, config
from rbtr.daemon.client import (
    _status,
    is_daemon_running,
    ping_daemon,
    start_daemon,
    stop_daemon,
    try_daemon,
)
from rbtr.daemon.messages import (
    BuildIndexRequest,
    BuildIndexResponse,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    FindRefsRequest,
    FindRefsResponse,
    GcMode,
    GcRequest,
    GcResponse,
    ListSymbolsRequest,
    ListSymbolsResponse,
    Notification,
    OkResponse,
    PingResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    Request,
    Response,
    SearchRequest,
    SearchResponse,
    ShutdownRequest,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.server import DaemonServer
from rbtr.daemon.status import DaemonStatusReport
from rbtr.errors import RbtrError
from rbtr.git import changed_files, open_repo
from rbtr.index.gc import run_gc
from rbtr.index.orchestrator import build_index, update_index
from rbtr.index.store import IndexStore

log = logging.getLogger(__name__)

# ── Internal server entry point ───────────────────────────────────────


class DaemonServe(BaseModel):
    """Internal: run the daemon server (spawned by `daemon start`)."""

    def cli_cmd(self) -> None:
        sock_dir = config.home
        sock_dir.mkdir(parents=True, exist_ok=True)
        store = IndexStore.from_config()
        server = DaemonServer(sock_dir, store)
        asyncio.run(server.serve())


# ── Daemon subcommands ───────────────────────────────────────────────


class DaemonStart(BaseModel):
    """Start the daemon in the background."""

    def cli_cmd(self) -> None:
        if is_daemon_running():
            s = _status()
            print_err(f"[yellow]Daemon already running (PID {s.pid if s else '?'}).[/]")
            return

        try:
            start_daemon()
        except RuntimeError as exc:
            print_err(f"[red]error:[/] {exc}")
            sys.exit(1)

        emit(OkResponse())


class DaemonStop(BaseModel):
    """Stop the daemon gracefully."""

    def cli_cmd(self) -> None:
        if not is_daemon_running():
            print_err("[yellow]Daemon is not running.[/]")
            return

        try:
            stop_daemon()
        except RuntimeError as exc:
            print_err(f"[red]error:[/] {exc}")
            sys.exit(1)

        emit(OkResponse())


class DaemonStatusCmd(BaseModel):
    """Check daemon status.

    Answers "is the daemon process running?" — distinct from
    ``rbtr status`` which answers "does an index exist for this
    repo?".
    """

    def cli_cmd(self) -> None:
        status = _status()
        if status is None or not is_daemon_running():
            emit(DaemonStatusReport(running=False))
            return

        t0 = time.monotonic()
        ping = ping_daemon()
        ping_ms = (time.monotonic() - t0) * 1000.0 if ping is not None else None

        emit(
            DaemonStatusReport(
                running=ping is not None,
                pid=status.pid,
                rpc=status.rpc,
                pub=status.pub,
                version=ping.version if ping is not None else status.version,
                uptime_seconds=ping.uptime if ping is not None else None,
                ping_ms=ping_ms,
            )
        )


class Daemon(BaseModel):
    """Manage the rbtr daemon."""

    start: CliSubCommand[DaemonStart]
    stop: CliSubCommand[DaemonStop]
    status: CliSubCommand[DaemonStatusCmd]
    serve: CliSubCommand[DaemonServe]

    def cli_cmd(self) -> None:
        # config.json_output is already set by root Rbtr.cli_cmd()
        CliApp.run_subcommand(self)


# ── Index subcommand ─────────────────────────────────────────────────


class Index(BaseModel):
    """Index a repository (one ref or base + head)."""

    refs: CliPositionalArg[list[str]] = Field(["HEAD"], description="Refs to index")
    repo_path: str = Field(".", description="Repository path")
    daemon: bool = Field(True, description="Use the daemon (disable with --no-daemon)")

    def cli_cmd(self) -> None:
        resolved_repo = str(Path(self.repo_path).resolve())
        repo = open_repo(resolved_repo)

        # Resolve refs to SHAs
        from rbtr.git import resolve_commit

        resolved_refs = [str(resolve_commit(repo, r).id) for r in self.refs]

        if not self.daemon:
            self._run_inline(resolved_repo, resolved_refs)
            return

        # Try daemon first
        resp = try_daemon(
            BuildIndexRequest(repo=resolved_repo, refs=resolved_refs)
        )
        if resp is not None:
            if isinstance(resp, BuildIndexResponse):
                emit(resp)
            else:
                print_err(f"[red]error:[/] unexpected response: {resp}")
            return

        # Daemon not running: auto-start and retry
        try:
            start_daemon()
        except RuntimeError as exc:
            print_err(f"[red]error:[/] failed to start daemon: {exc}")
            print_err("[dim]Falling back to inline execution.[/]")
            self._run_inline(resolved_repo, resolved_refs)
            return

        resp = try_daemon(
            BuildIndexRequest(repo=resolved_repo, refs=resolved_refs)
        )
        if resp is not None and isinstance(resp, BuildIndexResponse):
            emit(resp)
        else:
            print_err("[yellow]Index job queued (daemon started).[/]")
            print_err("[dim]Run `rbtr daemon status` to track progress.[/]")

    def _run_inline(self, resolved_repo: str, resolved_refs: list[str]) -> None:
        """Run indexing inline (blocking)."""
        repo = open_repo(resolved_repo)
        store = IndexStore.from_config()

        with progress_reporter("Parsing files", "Embedding") as (on_parse, on_embed):
            if len(resolved_refs) == 2:
                build_index(
                    repo,
                    resolved_refs[0],
                    store,
                    on_progress=on_parse,
                    on_embed_progress=on_embed,
                )
                result = update_index(
                    repo,
                    resolved_refs[0],
                    resolved_refs[1],
                    store,
                    on_progress=on_parse,
                    on_embed_progress=on_embed,
                )
            else:
                result = build_index(
                    repo,
                    resolved_refs[0],
                    store,
                    on_progress=on_parse,
                    on_embed_progress=on_embed,
                )

        emit(
            BuildIndexResponse(
                refs=resolved_refs,
                stats=result.stats,
                errors=result.errors,
            )
        )


# ── Read subcommands (daemon-first, fallback) ───────────────────────


class Search(BaseModel):
    """Search the code index."""

    query: CliPositionalArg[str] = Field(description="Search query")
    limit: int = Field(10, description="Maximum results to return")
    ref: str = Field("HEAD", description="Git ref (for diff proximity scoring)")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = str(Path(self.repo_path).resolve())
        resp = try_daemon(
            SearchRequest(repo=resolved_repo, query=self.query, limit=self.limit, ref=self.ref)
        )
        if isinstance(resp, SearchResponse):
            for r in resp.results:
                emit(r)
            return

        # Fallback: direct execution
        store = IndexStore.from_config()
        repo_id = store.register_repo(resolved_repo)
        for r in store.search(self.ref, self.query, top_k=self.limit, repo_id=repo_id):
            emit(r)


class ReadSymbol(BaseModel):
    """Read a symbol's full source."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name (e.g. HttpClient.retry)")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = str(Path(self.repo_path).resolve())
        resp = try_daemon(
            ReadSymbolRequest(repo=resolved_repo, name=self.symbol, ref=self.ref)
        )
        if isinstance(resp, ReadSymbolResponse):
            for c in resp.chunks:
                emit(c)
            return

        # Fallback: direct execution
        store = IndexStore.from_config()
        repo_id = store.register_repo(resolved_repo)
        chunks = store.search_by_name(self.ref, self.symbol, repo_id=repo_id)
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
        resolved_repo = str(Path(self.repo_path).resolve())
        resp = try_daemon(
            ListSymbolsRequest(repo=resolved_repo, file_path=self.file, ref=self.ref)
        )
        if isinstance(resp, ListSymbolsResponse):
            for c in resp.chunks:
                emit(c, compact=True)
            return

        # Fallback: direct execution
        store = IndexStore.from_config()
        repo_id = store.register_repo(resolved_repo)
        for c in store.get_chunks(self.ref, file_path=self.file, repo_id=repo_id):
            emit(c, compact=True)


class FindRefs(BaseModel):
    """Find references to a symbol via the dependency graph."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name")
    ref: str = Field("HEAD", description="Git ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = str(Path(self.repo_path).resolve())
        resp = try_daemon(
            FindRefsRequest(repo=resolved_repo, symbol=self.symbol, ref=self.ref)
        )
        if isinstance(resp, FindRefsResponse):
            for e in resp.edges:
                emit(e)
            return

        # Fallback: direct execution
        store = IndexStore.from_config()
        repo_id = store.register_repo(resolved_repo)
        for e in store.get_edges(self.ref, target_id=self.symbol, repo_id=repo_id):
            emit(e)


class ChangedSymbols(BaseModel):
    """Show symbols that changed between two refs."""

    base: str = Field(description="Base ref")
    head: str = Field(description="Head ref")
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = str(Path(self.repo_path).resolve())
        resp = try_daemon(
            ChangedSymbolsRequest(repo=resolved_repo, base=self.base, head=self.head)
        )
        if isinstance(resp, ChangedSymbolsResponse):
            for c in resp.chunks:
                emit(c, compact=True)
            return

        # Fallback: direct execution
        repo = open_repo(resolved_repo)
        changed = changed_files(repo, self.base, self.head)
        store = IndexStore.from_config()
        repo_id = store.register_repo(resolved_repo)
        for path in sorted(changed):
            for c in store.get_chunks(self.head, file_path=path, repo_id=repo_id):
                emit(c, compact=True)


class Status(BaseModel):
    """Show index status."""

    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = str(Path(self.repo_path).resolve())
        resp = try_daemon(StatusRequest(repo=resolved_repo))
        if isinstance(resp, StatusResponse):
            emit(resp)
            return

        # Fallback: direct execution
        db = config.db_path
        if not db.exists():
            emit(StatusResponse(exists=False))
            return
        store = IndexStore(db)
        repo_id = store.register_repo(resolved_repo)
        count = store.count_chunks("HEAD", repo_id=repo_id)
        indexed_refs = [sha for sha, _ in store.list_indexed_commits(repo_id)]
        emit(
            StatusResponse(
                exists=count > 0,
                db_path=str(db),
                total_chunks=count,
                indexed_refs=indexed_refs,
            )
        )


class Gc(BaseModel):
    """Garbage-collect a repository's index.

    By default drops every indexed commit except the current HEAD
    and sweeps crashed-build residue. Alternate modes let you keep
    specific refs, drop specific refs, or sweep residue only.
    """

    repo_path: str = Field(".", description="Repository path")
    keep_head_only: bool = Field(
        False, description="Keep only HEAD; default behaviour"
    )
    keep_refs: bool = Field(
        False,
        description="Keep HEAD, local branches, tags, and notes",
    )
    keep: CliPositionalArg[list[str]] = Field(
        [],
        description="Keep these refs (implicitly keeps HEAD); drops the rest",
    )
    drop: list[str] = Field(
        [],
        description="Drop these refs; mutually exclusive with --keep",
    )
    orphans: bool = Field(
        False,
        description="Sweep crashed-build residue only; no commits dropped",
    )
    dry_run: bool = Field(
        False, description="Report what would be removed without writing"
    )

    def cli_cmd(self) -> None:
        mode, refs = self._resolve_mode()
        resolved_repo = str(Path(self.repo_path).resolve())

        resp = try_daemon(
            GcRequest(
                repo=resolved_repo,
                mode=mode,
                refs=refs,
                dry_run=self.dry_run,
            )
        )
        if isinstance(resp, GcResponse):
            emit(resp)
            return

        # Fallback: direct execution.
        repo = open_repo(resolved_repo)
        store = IndexStore.from_config()
        repo_id = store.register_repo(resolved_repo)

        t0 = time.monotonic()
        try:
            counts = run_gc(
                store,
                repo,
                repo_id,
                mode=mode,
                refs=refs,
                dry_run=self.dry_run,
            )
        except RbtrError as exc:
            print_err(f"[red]error:[/] {exc}")
            sys.exit(1)

        emit(
            GcResponse(
                commits_dropped=counts.commits,
                snapshots_dropped=counts.snapshots,
                edges_dropped=counts.edges,
                chunks_dropped=counts.chunks,
                elapsed_seconds=time.monotonic() - t0,
                dry_run=self.dry_run,
            )
        )

    def _resolve_mode(self) -> tuple[GcMode, list[str]]:
        """Pick the mode from the set of flags; enforce exclusivity."""
        if self.keep and self.drop:
            print_err("[red]error:[/] --keep and --drop are mutually exclusive")
            sys.exit(2)
        if self.orphans:
            return GcMode.ORPHANS, []
        if self.drop:
            return GcMode.DROP, self.drop
        if self.keep:
            return GcMode.KEEP, self.keep
        if self.keep_refs:
            return GcMode.KEEP_REFS, []
        # keep_head_only is the default whether or not the flag is set
        return GcMode.HEAD_ONLY, []


class SchemaDump(BaseModel):
    """Dump JSON Schema for the daemon protocol.

    Useful for generating TypeScript types or validating messages.
    """

    def cli_cmd(self) -> None:
        # Always outputs JSON — ignore json_output mode.
        out = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "rbtr Protocol",
            "request": TypeAdapter(Request).json_schema(),
            "response": TypeAdapter(Response).json_schema(),
            "notification": TypeAdapter(Notification).json_schema(),
        }
        print(json.dumps(out, indent=2))


# ── Root command ─────────────────────────────────────────────────────


class Rbtr(
    Config,
    cli_prog_name="rbtr",
    cli_kebab_case=True,
    cli_implicit_flags=True,
):
    """rbtr — structural code index."""

    daemon: CliSubCommand[Daemon]
    index: CliSubCommand[Index]
    search: CliSubCommand[Search]
    read_symbol: CliSubCommand[ReadSymbol]
    list_symbols: CliSubCommand[ListSymbols]
    find_refs: CliSubCommand[FindRefs]
    changed_symbols: CliSubCommand[ChangedSymbols]
    status: CliSubCommand[Status]
    gc: CliSubCommand[Gc]
    schema_dump: CliSubCommand[SchemaDump]

    def cli_cmd(self) -> None:
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
