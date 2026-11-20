"""CLI entry point for rbtr — structural code index.

Uses pydantic-settings CliApp for subcommand parsing. Config is
loaded from TOML/env, subcommand args from the CLI.

Output modes:
- **TTY** (interactive): rich-formatted text with syntax
  highlighting, coloured scores, and progress bars.
- **Piped / --json**: a single JSON response object to stdout
  (the same shape the daemon returns).

Both modes emit the **same data** — the pydantic output models
are the single source of truth. The human format is a richer
layout of the same fields; it never drops or adds information.

Dual-mode: most commands try the daemon first; if unreachable,
fall back to direct in-process execution. `rbtr watch` auto-starts
the daemon unless `--no-daemon` is given.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from enum import StrEnum
from typing import Annotated, Self

import duckdb
import structlog
from pydantic import BaseModel, BeforeValidator, Field, ValidationError, model_validator
from pydantic_settings import (
    CliApp,
    CliPositionalArg,
    CliSettingsSource,
    CliSubCommand,
    get_subcommand,
)
from rich_argparse import RichHelpFormatter

from rbtr.cli.output import (
    emit,
    fail,
    print_banner,
    print_err,
    print_json_schema,
    progress_reporter,
    reject_arguments,
)
from rbtr.config import Config, WeightTriple, config
from rbtr.daemon.client import (
    live_status,
    start_daemon,
    stop_daemon,
    try_daemon,
)
from rbtr.daemon.handlers import (
    handle_changed_symbols,
    handle_daemon_config,
    handle_find_refs,
    handle_forget,
    handle_forget_stale,
    handle_gc,
    handle_gc_all,
    handle_list_symbols,
    handle_read_symbol,
    handle_search,
    handle_search_all,
    handle_status,
    handle_status_all,
    handle_unwatch,
    handle_unwatch_stale,
    handle_unwatch_stale_all,
)
from rbtr.daemon.messages import (
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    DaemonConfigRequest,
    DaemonConfigResponse,
    ErrorCode,
    ErrorResponse,
    ExcludeKinds,
    FindRefsRequest,
    FindRefsResponse,
    ForgetRequest,
    ForgetResponse,
    ForgetStaleRequest,
    GcAllRequest,
    GcRequest,
    GcResponse,
    IncludeKinds,
    IndexState,
    KindFilter,
    ListSymbolsRequest,
    ListSymbolsResponse,
    OkResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    SearchAllRequest,
    SearchRequest,
    SearchResponse,
    StatusAllRequest,
    StatusRequest,
    StatusResponse,
    UnwatchRequest,
    UnwatchResponse,
    UnwatchStaleAllRequest,
    UnwatchStaleRequest,
    WatchRequest,
    WatchResponse,
    WatchSetResponse,
    protocol_json_schema,
)
from rbtr.daemon.server import Daemon
from rbtr.daemon.status import DaemonStatusReport, uptime_seconds as _uptime_seconds
from rbtr.domain.models import ChunkKind, GcMode, Scope
from rbtr.errors import ExitCode, IndexLockedError, IndexNotBuiltError, RbtrError
from rbtr.git import HEAD_REF, normalise_repo_path, resolve_ref
from rbtr.index.build import build_index
from rbtr.index.embed import embed_index
from rbtr.index.embeddings import Embedder
from rbtr.index.index import Index
from rbtr.index.reranker import Reranker
from rbtr.languages.manager import get_manager
from rbtr.logging import configure_logging

log = structlog.get_logger(__name__)


def _normalise_choice(value: str | Scope | GcMode) -> str | Scope | GcMode:
    """Accept a choice as typed: any case, dashes or underscores."""
    return value.lower().replace("-", "_") if isinstance(value, str) else value


# Case-insensitive choices: a human may type `all`, `ALL`, or `All`,
# and `watched-only` for `watched_only`.  Normalisation happens here
# at the CLI boundary; the wire protocol stays strict.
ScopeField = Annotated[Scope, BeforeValidator(_normalise_choice)]


class KeptSet(StrEnum):
    """The sets `--keep` can name, each a `GcMode` it selects.

    `GcMode.KEEP` has no member here: the refs it keeps are named by
    `--keep-refs`, which is what selects it.  `EVERYTHING` is
    `GcMode.ORPHANS` said as what survives it — every commit, since
    it sweeps crashed-build residue alone.
    """

    WATCHED = "watched"
    WATCHED_ONLY = "watched_only"
    HEAD_ONLY = "head_only"
    EVERYTHING = "everything"


KeepField = Annotated[KeptSet, BeforeValidator(_normalise_choice)]


class DaemonServe(BaseModel):
    """Internal: run the daemon server (spawned by `daemon start`)."""

    allow_missing_plugins: bool = Field(
        False,
        description="Serve even if the index has languages this rbtr cannot load",
    )

    def cli_cmd(self) -> None:
        # The daemon's sole log sink is the rotating JSON file; the
        # parent no longer redirects the child's stderr here.
        configure_logging(to_file=True)

        config.runtime_dir.mkdir(parents=True, exist_ok=True)
        print_banner()
        try:
            index = Index.from_config(writable=True)
        except IndexLockedError:
            # Another process holds the index; the store logged it.  The
            # code tells `start_daemon` to wait for that process's daemon.
            sys.exit(ExitCode.INDEX_LOCKED)
        except RbtrError:
            # The parent redirects our stderr to DEVNULL, so record why
            # the daemon couldn't open the index (e.g. a schema refuse)
            # in the log the parent points the user at.
            log.exception("daemon_store_open_failed")
            raise
        server = Daemon(config.runtime_dir, index, allow_missing_plugins=self.allow_missing_plugins)
        asyncio.run(server.serve())


class DaemonStart(BaseModel):
    """Start the daemon in the background."""

    allow_missing_plugins: bool = Field(
        False,
        description="Serve even if the index has languages this rbtr cannot load",
    )

    def cli_cmd(self) -> None:
        running = live_status(config.runtime_dir)
        if running is not None:
            print_err(f"[yellow]Daemon already running (PID {running.pid}).[/]")
        else:
            try:
                running = start_daemon(allow_missing_plugins=self.allow_missing_plugins)
            except RbtrError as exc:
                fail(exc)

        emit(
            DaemonStatusReport(
                running=True,
                pid=running.pid,
                rpc=running.rpc,
                pub=running.pub,
                version=running.version,
                uptime_seconds=_uptime_seconds(running.started_at),
            )
        )


class DaemonStop(BaseModel):
    """Stop the daemon gracefully."""

    def cli_cmd(self) -> None:
        if live_status(config.runtime_dir) is None:
            print_err("[yellow]Daemon is not running.[/]")
            return

        try:
            stop_daemon()
        except RbtrError as exc:
            fail(exc)

        emit(OkResponse())


class DaemonStatusCmd(BaseModel):
    """Check daemon status.

    Answers "is the daemon process running?" — distinct from
    `rbtr status` which answers "does an index exist for this
    repo?".
    """

    def cli_cmd(self) -> None:
        status = live_status(config.runtime_dir)
        if status is None:
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


class DaemonCmd(BaseModel):
    """Manage the rbtr daemon."""

    start: CliSubCommand[DaemonStart]
    stop: CliSubCommand[DaemonStop]
    status: CliSubCommand[DaemonStatusCmd]
    serve: CliSubCommand[DaemonServe]

    def cli_cmd(self) -> None:
        # config.json_output is already set by root Rbtr.cli_cmd()
        CliApp.run_subcommand(self)


class Watch(BaseModel):
    """Watch refs and keep them indexed.

    Each positional ref is an independent watch target the daemon
    keeps indexed; `rbtr watch` with no args watches `HEAD`.
    Stop watching one with `rbtr unwatch`.
    """

    refs: CliPositionalArg[list[str]] = Field([HEAD_REF], description="Refs to watch and index")
    repo_path: str = Field(".", description="Repository path")
    daemon: bool = Field(True, description="Use the daemon (disable with --no-daemon)")
    allow_missing_plugins: bool = Field(
        False,
        description="Build even if the index has languages this rbtr cannot load",
    )

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)

        # Validate every ref (raises on a bad ref); the daemon stores
        # symbolic names so moving refs track their tip.
        for r in self.refs:
            resolve_ref(resolved_repo, r, max_untracked_files=config.max_untracked_files)

        request = WatchRequest(repo_path=resolved_repo, refs=self.refs)

        if not self.daemon:
            self._run_inline(
                resolved_repo,
                [
                    resolve_ref(resolved_repo, r, max_untracked_files=config.max_untracked_files)
                    for r in self.refs
                ],
            )
            return

        # Try daemon first
        resp = try_daemon(request)
        if resp is not None:
            self._report_watch(resp, started=False)
            return

        # Daemon not running: auto-start and retry
        try:
            start_daemon()
        except RbtrError as exc:
            print_err(f"[red]error:[/] failed to start daemon: {exc}")
            # No "falling back" notice before the attempt: when the start
            # failed on the DuckDB lock, the inline build cannot run
            # either, and announcing a fallback that can't happen is how
            # this failure reads as a mystery.  `_run_inline` opens the
            # store as its first act, so the lock surfaces immediately.
            try:
                self._run_inline(
                    resolved_repo,
                    [
                        resolve_ref(
                            resolved_repo, r, max_untracked_files=config.max_untracked_files
                        )
                        for r in self.refs
                    ],
                )
            except IndexLockedError as locked:
                message = f"{locked} Inline indexing needs that same lock, so it was not attempted."
                fail(ErrorResponse(code=ErrorCode(locked.error_code), message=message))
            return

        self._report_watch(try_daemon(request), started=True)

    def _report_watch(self, resp: object, *, started: bool) -> None:
        """Print the outcome of a daemon watch (add) request."""
        suffix = " (daemon started)" if started else ""
        match resp:
            case WatchResponse():
                emit(resp)
            case WatchSetResponse():
                emit(resp)
                print_err(f"[dim]Indexing in background{suffix}; run `rbtr status` to track.[/]")
            case ErrorResponse() as err:
                fail(err)
            case _:
                fail(
                    ErrorResponse(
                        code=ErrorCode.INTERNAL, message="daemon did not record the watch"
                    )
                )

    def _run_inline(
        self,
        resolved_repo: str,
        resolved_refs: list[str],
    ) -> None:
        """Run indexing inline (blocking)."""
        index = Index.from_config(writable=True)
        # Refuse to rebuild languages whose plugin isn't loaded -- that
        # would re-extract raw chunks and churn embeddings -- unless the
        # caller overrides with --allow-missing-plugins.
        get_manager().require_languages_loaded(
            index.distinct_chunk_languages(), allow_missing=self.allow_missing_plugins
        )
        repo = index.register(resolved_repo)

        with progress_reporter("Parsing files", "Embedding") as (on_parse, on_embed):

            def on_progress(phase: str, done: int, total: int) -> None:
                if phase == "embedding":
                    on_embed(done, total)
                elif phase == "parsing":
                    on_parse(done, total)

            # Each ref is an independent full build.
            result = build_index(
                resolved_repo,
                resolved_refs[0],
                index,
                on_progress=on_progress,
            )
            for ref in resolved_refs[1:]:
                result = build_index(
                    resolved_repo,
                    ref,
                    index,
                    on_progress=on_progress,
                )

            # Embed all refs after chunks/edges are committed.
            if config.embed:
                embedder = Embedder(idle_timeout=config.embed_idle_timeout)
                for ref in resolved_refs:
                    embedded = embed_index(
                        index,
                        at=repo.at(ref),
                        embedder=embedder,
                        on_progress=on_progress,
                    )
                    result.stats.embedded_chunks += embedded

        emit(
            WatchResponse(
                resolved_refs=resolved_refs,
                stats=result.stats,
                errors=result.errors,
            )
        )


class Unwatch(BaseModel):
    """Stop watching refs.

    Names the refs to drop, or finds the ones git can no longer
    resolve with `--stale`.  The index they built stays until
    `rbtr gc --keep watched-only` reclaims it.
    """

    refs: CliPositionalArg[list[str]] = Field([], description="Refs to stop watching")
    repo_path: str = Field(".", description="Repository path")
    stale: bool = Field(
        False,
        description="Stop watching refs that no longer resolve (deleted branches)",
    )
    scope: ScopeField = Field(
        Scope.WORKSPACE,
        description="Unwatch scope: workspace (this repo) or all (every indexed repo).",
    )
    dry_run: bool = Field(False, description="Report what would be unwatched")
    daemon: bool = Field(True, description="Use the daemon (disable with --no-daemon)")

    @model_validator(mode="after")
    def _check_refs(self) -> Self:
        """A run drops one set of refs, chosen one way."""
        if self.stale and self.refs:
            msg = "--stale finds the refs to drop, so naming refs as well says it twice"
            raise ValueError(msg)
        if not self.stale and not self.refs:
            msg = "name the refs to stop watching, or --stale to drop the ones git has lost"
            raise ValueError(msg)
        if self.scope is Scope.ALL and not self.stale:
            msg = "a named ref is watched in one repo; every repo answers only for --stale"
            raise ValueError(msg)
        return self

    def cli_cmd(self) -> None:
        request: UnwatchRequest | UnwatchStaleRequest | UnwatchStaleAllRequest
        if self.scope is Scope.ALL:
            request = UnwatchStaleAllRequest(dry_run=self.dry_run)
        elif self.stale:
            request = UnwatchStaleRequest(
                repo_path=normalise_repo_path(self.repo_path), dry_run=self.dry_run
            )
        else:
            request = UnwatchRequest(
                repo_path=normalise_repo_path(self.repo_path), refs=self.refs, dry_run=self.dry_run
            )
        resp = try_daemon(request) if self.daemon else None
        if resp is None:
            index = Index.from_config(writable=True)
            try:
                match request:
                    case UnwatchStaleAllRequest():
                        resp = handle_unwatch_stale_all(request, index)
                    case UnwatchStaleRequest():
                        resp = handle_unwatch_stale(request, index)
                    case UnwatchRequest():
                        resp = handle_unwatch(request, index)
            finally:
                index.close()
        match resp:
            case UnwatchResponse():
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case _:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class Forget(BaseModel):
    """Forget a repo's index: this one, or the ones whose checkout is gone.

    Forgetting drops the watch set, indexed commits and references.
    It is metadata-only — run `rbtr gc` to reclaim the space.
    """

    repo_path: str = Field(".", description="Repository path")
    stale: bool = Field(
        False,
        description="Forget every indexed repo whose path no longer exists (removed checkouts)",
    )
    dry_run: bool = Field(False, description="Report what would be forgotten")
    daemon: bool = Field(True, description="Use the daemon (disable with --no-daemon)")

    @model_validator(mode="after")
    def _check_repo_path(self) -> Self:
        """A repo that is gone cannot also be the repo you name.

        `model_fields_set` tells a path the caller passed from the
        default that happens to match it.
        """
        if self.stale and "repo_path" in self.model_fields_set:
            msg = "the repos whose checkout is gone are found, not named by path"
            raise ValueError(msg)
        return self

    def cli_cmd(self) -> None:
        # A vanished checkout cannot be named, so the stale request carries
        # no repo path and runs from anywhere, including outside a git repo.
        request: ForgetRequest | ForgetStaleRequest = (
            ForgetStaleRequest(dry_run=self.dry_run)
            if self.stale
            else ForgetRequest(repo_path=normalise_repo_path(self.repo_path), dry_run=self.dry_run)
        )
        resp = try_daemon(request) if self.daemon else None
        if resp is None:
            index = Index.from_config(writable=True)
            try:
                resp = (
                    handle_forget_stale(request, index)
                    if isinstance(request, ForgetStaleRequest)
                    else handle_forget(request, index)
                )
            finally:
                index.close()
        match resp:
            case ForgetResponse():
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case _:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


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
    repo_path: str = Field(".", description="Repository path")
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
    keywords: list[str] | None = Field(
        None, description="Extra keyword to widen retrieval (repeatable)"
    )
    variants: list[str] | None = Field(
        None, description="Alternative phrasing of the query (repeatable)"
    )
    scope: ScopeField = Field(
        Scope.WORKSPACE,
        description="Search scope: workspace (this repo) or all (every indexed repo).",
    )
    preview_lines: int | None = Field(
        None, ge=1, description="Lines of each hit's body to preview (default: the config's)"
    )

    @model_validator(mode="after")
    def _check_ref(self) -> Self:
        """A ref is one repo's: every repo is searched at its latest snapshot."""
        if self.scope is Scope.ALL and self.ref is not None:
            msg = "--ref names one repo's snapshot; --scope all searches each repo's latest"
            raise ValueError(msg)
        return self

    def cli_cmd(self) -> None:
        weights = None
        if self.alpha is not None and self.beta is not None and self.gamma is not None:
            weights = WeightTriple(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        request: SearchRequest | SearchAllRequest
        if self.scope is Scope.ALL:
            request = SearchAllRequest(
                query=self.query,
                limit=self.limit,
                weights=weights,
                query_kind=self.query_kind,
                keywords=self.keywords,
                variants=self.variants,
                preview_lines=self.preview_lines,
            )
        else:
            request = SearchRequest(
                repo_path=normalise_repo_path(self.repo_path),
                ref=self.ref,
                query=self.query,
                limit=self.limit,
                weights=weights,
                query_kind=self.query_kind,
                keywords=self.keywords,
                variants=self.variants,
                preview_lines=self.preview_lines,
            )

        match try_daemon(request):
            case SearchResponse() as resp:
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case None:
                index = Index.from_config()

                gpu_load_lock = threading.Lock()
                embedder = Embedder(load_lock=gpu_load_lock) if config.embed else None
                reranker = Reranker(load_lock=gpu_load_lock) if config.reranker_model else None
                try:
                    match request:
                        case SearchAllRequest():
                            resp = handle_search_all(
                                request, index, embedder=embedder, reranker=reranker
                            )
                        case SearchRequest():
                            resp = handle_search(
                                request, index, embedder=embedder, reranker=reranker
                            )
                    emit(resp)
                except RbtrError as exc:
                    fail(exc)
                finally:
                    if embedder is not None:
                        embedder.close()
                    if reranker is not None:
                        reranker.close()
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class ReadSymbol(BaseModel):
    """Read a symbol's full source."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name (e.g. HttpClient.retry)")
    ref: str | None = Field(
        None, description="Git ref (defaults to working tree if dirty, HEAD if clean)"
    )
    repo_path: str = Field(".", description="Repository path")
    file_path: list[str] | None = Field(
        None, description="Limit to symbols in these files (repeatable)"
    )

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)
        request = ReadSymbolRequest(
            repo_path=resolved_repo,
            symbol=self.symbol,
            ref=self.ref,
            file_paths=self.file_path,
        )

        match try_daemon(request):
            case ReadSymbolResponse() as resp:
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case None:
                index = Index.from_config()

                try:
                    emit(handle_read_symbol(request, index))
                except RbtrError as exc:
                    fail(exc)
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class ListSymbols(BaseModel):
    """List symbols in a file (table of contents)."""

    file: CliPositionalArg[str] = Field(description="File path")
    ref: str | None = Field(
        None, description="Git ref (defaults to working tree if dirty, HEAD if clean)"
    )
    repo_path: str = Field(".", description="Repository path")
    kind: list[ChunkKind] | None = Field(
        None, description="Keep only symbols of this kind (repeatable)"
    )
    exclude_kind: list[ChunkKind] | None = Field(
        None, description="Drop symbols of this kind (repeatable)"
    )

    @model_validator(mode="after")
    def _check_kinds(self) -> Self:
        """The request takes one list of kinds, kept or dropped."""
        if self.kind and self.exclude_kind:
            msg = "--kind keeps kinds and --exclude-kind drops them; use one"
            raise ValueError(msg)
        return self

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)
        kinds: KindFilter | None = None
        if self.kind:
            kinds = IncludeKinds(include=self.kind)
        elif self.exclude_kind:
            kinds = ExcludeKinds(exclude=self.exclude_kind)
        request = ListSymbolsRequest(
            repo_path=resolved_repo,
            file_path=self.file,
            ref=self.ref,
            kinds=kinds,
        )

        match try_daemon(request):
            case ListSymbolsResponse() as resp:
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case None:
                index = Index.from_config()

                try:
                    emit(handle_list_symbols(request, index))
                except RbtrError as exc:
                    fail(exc)
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class FindRefs(BaseModel):
    """Find references to a symbol via the dependency graph."""

    symbol: CliPositionalArg[str] = Field(description="Symbol name")
    ref: str | None = Field(
        None, description="Git ref (defaults to working tree if dirty, HEAD if clean)"
    )
    repo_path: str = Field(".", description="Repository path")
    file_path: list[str] | None = Field(
        None, description="Limit to symbols in these files (repeatable)"
    )

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)
        request = FindRefsRequest(
            repo_path=resolved_repo,
            symbol=self.symbol,
            ref=self.ref,
            file_paths=self.file_path,
        )

        match try_daemon(request):
            case FindRefsResponse() as resp:
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case None:
                index = Index.from_config()

                try:
                    emit(handle_find_refs(request, index))
                except RbtrError as exc:
                    fail(exc)
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class ChangedSymbols(BaseModel):
    """Show symbols that changed between two refs."""

    base: CliPositionalArg[str] = Field(description="Base ref")
    head: CliPositionalArg[str] = Field(description="Head ref")
    repo_path: str = Field(".", description="Repository path")
    file_path: list[str] | None = Field(
        None, description="Limit to changes in these files (repeatable)"
    )

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)
        request = ChangedSymbolsRequest(
            repo_path=resolved_repo,
            base=self.base,
            head=self.head,
            file_paths=self.file_path,
        )

        match try_daemon(request):
            case ChangedSymbolsResponse() as resp:
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case None:
                index = Index.from_config()

                try:
                    emit(handle_changed_symbols(request, index))
                except RbtrError as exc:
                    fail(exc)
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class Status(BaseModel):
    """Show index status."""

    repo_path: str = Field(".", description="Repository path")
    scope: ScopeField = Field(
        Scope.WORKSPACE,
        description="Status scope: workspace (this repo) or all (every indexed repo).",
    )

    def cli_cmd(self) -> None:
        request = (
            StatusAllRequest()
            if self.scope is Scope.ALL
            else StatusRequest(repo_path=normalise_repo_path(self.repo_path))
        )

        match try_daemon(request):
            case StatusResponse() as resp:
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case None:
                print_err("[yellow]![/]  rbtr is not running")
                db = config.db_path
                # No index yet: the reply still names the repository.
                empty = StatusResponse(
                    state=IndexState.NO_INDEX,
                    repo_path=request.repo_path if isinstance(request, StatusRequest) else None,
                )
                if not db.exists():
                    emit(empty)
                    return
                try:
                    index = Index(db)
                except IndexNotBuiltError:
                    emit(empty)  # the file exists, the index does not
                    return

                try:
                    match request:
                        case StatusAllRequest():
                            emit(handle_status_all(request, index))
                        case StatusRequest():
                            emit(handle_status(request, index))
                except RbtrError as exc:
                    fail(exc)
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class Gc(BaseModel):
    """Garbage-collect a repository's index.

    Destructive and not undoable — permanently deletes indexed
    commits/chunks. Use --dry-run to preview first.

    `--keep` says what a run keeps: `watched` (the default — HEAD,
    local branches and tags, the watch set and the current worktree),
    `watched-only`, `head-only`, or `orphans`, which drops no commits
    and sweeps crashed-build residue. `--keep-refs <refs>` keeps those
    refs and HEAD instead. Every run sweeps that residue.

    Operates on the current repo by default; `--scope all` reclaims
    across every indexed repo at once, keeping `watched` or
    `watched-only` — what each repo can answer in its own terms.
    """

    repo_path: str = Field(".", description="Repository path")
    scope: ScopeField = Field(
        Scope.WORKSPACE,
        description="GC scope: workspace (this repo) or all (every indexed repo).",
    )
    keep: KeepField = Field(
        KeptSet.WATCHED,
        description=(
            "What a run keeps: watched (HEAD, local branches/tags and the watch "
            "set), watched-only, head-only, or everything (every commit; sweeps "
            "crashed-build residue alone)."
        ),
    )
    keep_refs: list[str] = Field(
        [],
        description="Keep these refs and HEAD, dropping the rest (comma-separated)",
    )
    dry_run: bool = Field(False, description="Report what would be removed without writing")
    compact: bool = Field(
        True,
        description="Rewrite the index to reclaim freed disk space (--no-compact to skip)",
    )

    @property
    def mode(self) -> GcMode:
        """The retention these flags name; refs to keep are their own."""
        if self.keep_refs:
            return GcMode.KEEP
        if self.keep is KeptSet.EVERYTHING:
            return GcMode.ORPHANS
        return GcMode(self.keep.value)

    @model_validator(mode="after")
    def _check_retention(self) -> Self:
        """A run keeps one set of refs, chosen in terms every repo has.

        `--keep-refs` names that set outright, so naming it as well as
        a set to keep says it twice. And what a repo keeps when the run
        covers them all can only be said in its own terms — its HEAD,
        its watch set — never as a list of refs one of them has.
        """
        if self.keep_refs and "keep" in self.model_fields_set:
            msg = f"--keep-refs is the set kept, so --keep {self.keep} says it twice"
            raise ValueError(msg)
        if self.scope is Scope.ALL and self.mode not in (GcMode.WATCHED, GcMode.WATCHED_ONLY):
            msg = (
                f"a run across every repo keeps each repo's HEAD and watch set; "
                f"{self.mode.value} decides what a single repo keeps"
            )
            raise ValueError(msg)
        return self

    def cli_cmd(self) -> None:
        request = (
            # `_check_retention` has refused every mode but the two `GcAllRequest` takes.
            GcAllRequest.model_validate(
                {"mode": self.mode, "dry_run": self.dry_run, "compact": self.compact}
            )
            if self.scope is Scope.ALL
            else GcRequest(
                repo_path=normalise_repo_path(self.repo_path),
                mode=self.mode,
                refs=self.keep_refs,
                dry_run=self.dry_run,
                compact=self.compact,
            )
        )

        match try_daemon(request):
            case GcResponse() as resp:
                emit(resp)
            case ErrorResponse() as err:
                fail(err)
            case None:
                # No daemon: this process holds DuckDB's exclusive lock, so
                # compaction's connection swap is safe here (allow_compact).
                index = Index.from_config(writable=True)

                try:
                    self._run_inline(request, index)
                except RbtrError as exc:
                    fail(exc)
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))

    @staticmethod
    def _run_inline(request: GcRequest | GcAllRequest, index: Index) -> None:
        """Collect in this process, reporting progress on stderr.

        A pass over many repos says nothing until it finishes; the bar
        goes to stderr, leaving stdout to the response.
        """
        match request:
            case GcRequest():
                emit(handle_gc(request, index, allow_compact=True))
            case GcAllRequest():
                with progress_reporter("Collecting repos") as (on_repo,):
                    response = handle_gc_all(
                        request,
                        index,
                        allow_compact=True,
                        on_progress=lambda _repo_path, done, total: on_repo(done, total),
                    )
                emit(response)


class SchemaDump(BaseModel):
    """Dump JSON Schema for the daemon protocol.

    Useful for generating TypeScript types or validating messages.
    """

    def cli_cmd(self) -> None:
        # Always outputs JSON — ignore json_output mode.
        print_json_schema(protocol_json_schema())


class ConfigCmd(BaseModel):
    """Show the rbtr configuration and loaded language plugins.

    Reports the running daemon's live config when one is up; otherwise
    the locally resolved config (what a daemon would load), noted on
    stderr.
    """

    def cli_cmd(self) -> None:
        match try_daemon(DaemonConfigRequest()):
            case DaemonConfigResponse() as resp:
                emit(resp)
            case None:
                print_err("[dim]daemon not running — showing local config[/]")
                emit(handle_daemon_config(DaemonConfigRequest()))
            case ErrorResponse() as err:
                fail(err)
            case resp:
                fail(ErrorResponse(code=ErrorCode.INTERNAL, message=f"unexpected response: {resp}"))


class Rbtr(
    Config,
    cli_prog_name="rbtr",
    cli_kebab_case=True,
    cli_implicit_flags=True,
):
    """rbtr — structural code index."""

    daemon: CliSubCommand[DaemonCmd]
    watch: CliSubCommand[Watch]
    unwatch: CliSubCommand[Unwatch]
    forget: CliSubCommand[Forget]
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

        # Re-apply logging now that CLI overrides (level, format, dirs)
        # are on `config`; `main()` configured it earlier with defaults.
        configure_logging()

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
    and `ValidationError` at the outer boundary, so subcommand bodies
    don't each have to do the same try/except dance.
    """
    configure_logging()
    cli_source: CliSettingsSource[Rbtr] = CliSettingsSource(Rbtr, formatter_class=RichHelpFormatter)
    try:
        CliApp.run(Rbtr, cli_settings_source=cli_source)
    except RbtrError as exc:
        fail(exc, ExitCode.ERROR)
    except duckdb.OutOfMemoryException as exc:
        fail(ErrorResponse(code=ErrorCode.OUT_OF_MEMORY, message=str(exc)), ExitCode.ERROR)
    except ValidationError as exc:
        reject_arguments(exc)
