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
import json
import logging
import sys
import threading
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, TypeAdapter, ValidationError
from pydantic_settings import (
    CliApp,
    CliPositionalArg,
    CliSettingsSource,
    CliSubCommand,
    get_subcommand,
)
from rich.table import Table
from rich_argparse import RichHelpFormatter

from rbtr.cli.output import _out, emit, print_banner, print_err, progress_reporter
from rbtr.config import Config, WeightTriple, config
from rbtr.daemon.client import (
    _status,
    is_daemon_running,
    start_daemon,
    stop_daemon,
    try_daemon,
)
from rbtr.daemon.handlers import (
    handle_changed_symbols,
    handle_find_refs,
    handle_gc,
    handle_list_symbols,
    handle_read_symbol,
    handle_search,
    handle_status,
)
from rbtr.daemon.messages import (
    BuildIndexRequest,
    BuildIndexResponse,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    ErrorResponse,
    FindRefsRequest,
    FindRefsResponse,
    GcMode,
    GcRequest,
    GcResponse,
    ListSymbolsRequest,
    ListSymbolsResponse,
    Notification,
    OkResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    Request,
    Response,
    Scope,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.server import DaemonServer
from rbtr.daemon.status import DaemonStatusReport, uptime_seconds as _uptime_seconds
from rbtr.errors import RbtrError
from rbtr.git import normalise_repo_path, resolve_ref
from rbtr.index.embeddings import Embedder
from rbtr.index.orchestrator import build_index, embed_index
from rbtr.index.reranker import Reranker
from rbtr.index.store import IndexStore

log = logging.getLogger(__name__)


def _normalise_scope(value: str | Scope) -> str | Scope:
    """Lower-case string scope input so the `--scope` flag is case-insensitive."""
    return value.lower() if isinstance(value, str) else value


# Case-insensitive `--scope`: a human may type `all`, `ALL`, or
# `All`.  Normalisation happens here at the CLI boundary; the wire
# protocol (`SearchRequest`/`StatusRequest`) stays strict.
ScopeField = Annotated[Scope, BeforeValidator(_normalise_scope)]


# ── Internal server entry point ───────────────────────────────────────


class DaemonServe(BaseModel):
    """Internal: run the daemon server (spawned by `daemon start`)."""

    def cli_cmd(self) -> None:
        # Reconfigure logging to write directly to the daemon log
        # file.  The parent redirects stderr to this file too, but
        # stderr is block-buffered when it points to a file.
        # FileHandler flushes after every record.
        logging.basicConfig(
            filename=str(config.daemon_log),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
            level=logging.INFO,
            force=True,
        )

        config.runtime_dir.mkdir(parents=True, exist_ok=True)
        print_banner()
        store = IndexStore.from_config(writable=True)
        server = DaemonServer(config.runtime_dir, store)
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
    `rbtr status` which answers "does an index exist for this
    repo?".
    """

    def cli_cmd(self) -> None:
        status = _status()
        if status is None or not is_daemon_running():
            emit(DaemonStatusReport(running=False))
            return
        emit(
            DaemonStatusReport(
                running=True,
                pid=status.pid,
                rpc=status.rpc,
                pub=status.pub,
                version=status.version,
                uptime_seconds=_uptime_seconds(status.started_at),
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
    path: str = Field(".", description="Repository path")
    daemon: bool = Field(True, description="Use the daemon (disable with --no-daemon)")
    embed: bool = Field(True, description="Compute embeddings (disable with --no-embed)")

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.path)

        # Resolve refs to SHAs
        resolved_refs = [resolve_ref(resolved_repo, r) for r in self.refs]

        if not self.daemon:
            self._run_inline(resolved_repo, resolved_refs)
            return

        request = BuildIndexRequest(
            path=resolved_repo,
            refs=resolved_refs,
            embed=self.embed,
        )

        # Try daemon first
        resp = try_daemon(request)
        if resp is not None:
            match resp:
                case BuildIndexResponse():
                    emit(resp)
                case OkResponse():
                    print_err("[yellow]Index job queued.[/]")
                    print_err("[dim]Run `rbtr daemon status` to track progress.[/]")
                case _:
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

        resp = try_daemon(request)
        if resp is not None and isinstance(resp, (BuildIndexResponse, OkResponse)):
            print_err("[yellow]Index job queued (daemon started).[/]")
            print_err("[dim]Run `rbtr daemon status` to track progress.[/]")
        else:
            print_err("[red]error:[/] daemon started but failed to queue index job")

    def _run_inline(
        self,
        resolved_repo: str,
        resolved_refs: list[str],
    ) -> None:
        """Run indexing inline (blocking)."""
        store = IndexStore.from_config(writable=True)
        with store.session() as ws:
            repo_id = ws.register_repo(resolved_repo)
        embedder = Embedder(idle_timeout=config.embed_idle_timeout)

        with progress_reporter("Parsing files", "Embedding") as (on_parse, on_embed):

            def on_progress(phase: str, done: int, total: int) -> None:
                if phase == "embedding":
                    on_embed(done, total)
                elif phase == "parsing":
                    on_parse(done, total)

            if len(resolved_refs) == 2:
                build_index(
                    resolved_repo,
                    resolved_refs[0],
                    store,
                    repo_id=repo_id,
                    on_progress=on_progress,
                )
                result = build_index(
                    resolved_repo,
                    resolved_refs[1],
                    store,
                    repo_id=repo_id,
                    base_sha=resolved_refs[0],
                    on_progress=on_progress,
                )
            else:
                result = build_index(
                    resolved_repo,
                    resolved_refs[0],
                    store,
                    repo_id=repo_id,
                    on_progress=on_progress,
                )

            # Embed all refs after chunks/edges are committed.
            if self.embed:
                for ref in resolved_refs:
                    embedded = embed_index(
                        store,
                        ref,
                        repo_id=repo_id,
                        embedder=embedder,
                        on_progress=on_progress,
                    )
                    result.stats.embedded_chunks += embedded

        emit(
            BuildIndexResponse(
                resolved_refs=resolved_refs,
                stats=result.stats,
                errors=result.errors,
            )
        )


# ── Read subcommands (daemon-first, fallback) ───────────────────────


class Search(BaseModel):
    """Search the code index.

    `--alpha` / `--beta` / `--gamma` override per-`QueryKind`
    fusion weights uniformly for the call.  All three must be
    supplied together, each in `[0, 1]`, summing to `1.0`.
    """

    query: CliPositionalArg[str] = Field(description="Search query")
    limit: int = Field(10, description="Maximum results to return")
    ref: str | None = Field(
        None, description="Git ref (defaults to working tree if dirty, HEAD if clean)"
    )
    path: str = Field(".", description="Repository path")
    alpha: float | None = Field(
        None, description="Override fusion weight for the semantic channel."
    )
    beta: float | None = Field(None, description="Override fusion weight for the lexical channel.")
    gamma: float | None = Field(None, description="Override fusion weight for the name channel.")
    query_kind: str | None = Field(
        None,
        description=(
            "Force expansion pipeline: "
            "concept (keywords + variants), "
            "identifier (keywords only), "
            "code (no expansion). "
            "Omit to use the heuristic."
        ),
    )
    scope: ScopeField = Field(
        Scope.WORKSPACE,
        description="Search scope: workspace (this repo) or all (every indexed repo).",
    )

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.path)
        weights = None
        if self.alpha is not None and self.beta is not None and self.gamma is not None:
            weights = WeightTriple(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        try:
            request = SearchRequest(
                path=resolved_repo,
                query=self.query,
                limit=self.limit,
                ref=self.ref,
                weights=weights,
                query_kind=self.query_kind,
                scope=self.scope,
            )
        except ValidationError as exc:
            for err in exc.errors():
                msg = err["msg"].removeprefix("Value error, ")
                print_err(f"[red]error:[/] {msg}")
            sys.exit(2)

        match try_daemon(request):
            case SearchResponse() as resp:
                emit(resp)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                gpu_load_lock = threading.Lock()
                embedder = Embedder(load_lock=gpu_load_lock)
                reranker = Reranker(load_lock=gpu_load_lock) if config.reranker_model else None
                try:
                    resp = handle_search(
                        request,
                        store,
                        embedder=embedder,
                        reranker=reranker,
                    )
                    emit(resp)
                except RbtrError as exc:
                    print_err(f"[red]error:[/] {exc}")
                    sys.exit(1)
                finally:
                    embedder.close()
                    if reranker is not None:
                        reranker.close()
            case resp:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)


class ReadSymbol(BaseModel):
    """Read a symbol's full source."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name (e.g. HttpClient.retry)")
    ref: str | None = Field(
        None, description="Git ref (defaults to working tree if dirty, HEAD if clean)"
    )
    path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.path)
        request = ReadSymbolRequest(
            path=resolved_repo,
            name=self.symbol,
            ref=self.ref,
        )

        match try_daemon(request):
            case ReadSymbolResponse() as resp:
                for c in resp.chunks:
                    emit(c)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                try:
                    for c in handle_read_symbol(request, store).chunks:
                        emit(c)
                except RbtrError as exc:
                    print_err(f"[red]error:[/] {exc}")
                    sys.exit(1)
            case resp:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)


class ListSymbols(BaseModel):
    """List symbols in a file (table of contents)."""

    file: CliPositionalArg[str] = Field(description="File path")
    ref: str | None = Field(
        None, description="Git ref (defaults to working tree if dirty, HEAD if clean)"
    )
    path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.path)
        request = ListSymbolsRequest(
            path=resolved_repo,
            file_path=self.file,
            ref=self.ref,
        )

        match try_daemon(request):
            case ListSymbolsResponse() as resp:
                for c in resp.chunks:
                    emit(c, compact=True)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                try:
                    for c in handle_list_symbols(request, store).chunks:
                        emit(c, compact=True)
                except RbtrError as exc:
                    print_err(f"[red]error:[/] {exc}")
                    sys.exit(1)
            case resp:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)


class FindRefs(BaseModel):
    """Find references to a symbol via the dependency graph."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name")
    ref: str | None = Field(
        None, description="Git ref (defaults to working tree if dirty, HEAD if clean)"
    )
    path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.path)
        request = FindRefsRequest(path=resolved_repo, symbol=self.symbol, ref=self.ref)

        match try_daemon(request):
            case FindRefsResponse() as resp:
                for e in resp.edges:
                    emit(e)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                try:
                    for e in handle_find_refs(request, store).edges:
                        emit(e)
                except RbtrError as exc:
                    print_err(f"[red]error:[/] {exc}")
                    sys.exit(1)
            case resp:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)


class ChangedSymbols(BaseModel):
    """Show symbols that changed between two refs."""

    base: CliPositionalArg[str] = Field(description="Base ref")
    head: CliPositionalArg[str] = Field(description="Head ref")
    path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.path)
        request = ChangedSymbolsRequest(
            path=resolved_repo,
            base=self.base,
            head=self.head,
        )

        match try_daemon(request):
            case ChangedSymbolsResponse() as resp:
                emit(resp)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                try:
                    emit(handle_changed_symbols(request, store))
                except RbtrError as exc:
                    print_err(f"[red]error:[/] {exc}")
                    sys.exit(1)
            case resp:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)


class Status(BaseModel):
    """Show index status."""

    path: str = Field(".", description="Repository path")
    scope: ScopeField = Field(
        Scope.WORKSPACE,
        description="Status scope: workspace (this repo) or all (every indexed repo).",
    )

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.path)
        request = StatusRequest(path=resolved_repo, scope=self.scope)

        match try_daemon(request):
            case StatusResponse() as resp:
                emit(resp)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                db = config.db_path
                if not db.exists():
                    emit(StatusResponse())
                    return
                store = IndexStore(db)

                try:
                    emit(handle_status(request, store))
                except RbtrError as exc:
                    print_err(f"[red]error:[/] {exc}")
                    sys.exit(1)
            case resp:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)


class Gc(BaseModel):
    """Garbage-collect a repository's index.

    By default drops every indexed commit except the current HEAD
    and sweeps crashed-build residue. Alternate modes let you keep
    specific refs, drop specific refs, or sweep residue only.
    """

    path: str = Field(".", description="Repository path")
    keep_head_only: bool = Field(False, description="Keep only HEAD; default behaviour")
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
    dry_run: bool = Field(False, description="Report what would be removed without writing")

    def cli_cmd(self) -> None:
        mode, refs = self._resolve_mode()
        resolved_repo = normalise_repo_path(self.path)
        request = GcRequest(
            path=resolved_repo,
            mode=mode,
            refs=refs,
            dry_run=self.dry_run,
        )

        match try_daemon(request):
            case GcResponse() as resp:
                emit(resp)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config(writable=True)

                try:
                    emit(handle_gc(request, store))
                except RbtrError as exc:
                    print_err(f"[red]error:[/] {exc}")
                    sys.exit(1)
            case resp:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)

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
            # CLI-only shapes — not part of any protocol union but
            # exported so TypeScript can deserialise the JSON that
            # `rbtr daemon status --json` prints.
            "cli": {
                "DaemonStatusReport": DaemonStatusReport.model_json_schema(),
            },
        }
        print(json.dumps(out, indent=2))


class ConfigCmd(BaseModel):
    """Show the rendered rbtr configuration."""

    def cli_cmd(self) -> None:
        if config.json_output or not sys.stdout.isatty():
            sys.stdout.write(config.model_dump_json(indent=2))
            sys.stdout.write("\n")
            return

        toml_path = config.config_dir / "config.toml"
        table = Table(title="rbtr configuration")
        table.add_column("Setting", style="bold")
        table.add_column("Value")
        for key, value in sorted(config.model_dump(mode="json").items()):
            table.add_row(key, str(value))
        _out.print(table)
        _out.print(
            f"[dim]config file:[/] {toml_path} {'(exists)' if toml_path.exists() else '(not found)'}"
        )


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
    config: CliSubCommand[ConfigCmd]

    def cli_cmd(self) -> None:
        # Top-level flags declared on `Config` are parsed into the
        # `Rbtr` CLI instance; propagate them to the module-global
        # `config` so code paths that read `from rbtr.config import
        # config` see the overrides.  Without this, CLI flags
        # silently fell back to defaults.

        for field in Config.model_fields:
            setattr(config, field, getattr(self, field))

        sub = get_subcommand(self, is_required=False)
        if sub is None:
            if sys.stderr.isatty():
                print_banner()
            CliApp.print_help(self)
            return
        CliApp.run_subcommand(self)


def main() -> None:
    """Entry point for the rbtr CLI.

    Catches `RbtrError` (and its subclasses, e.g. `DaemonBusyError`)
    at the outer boundary so subcommand bodies don't each have to
    do the same try/except dance.
    """
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        level=logging.INFO,
    )
    cli_source: CliSettingsSource[Rbtr] = CliSettingsSource(Rbtr, formatter_class=RichHelpFormatter)
    try:
        CliApp.run(Rbtr, cli_settings_source=cli_source)
    except RbtrError as exc:
        print_err(f"[red]error:[/] {exc}")
        sys.exit(2)
