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
fall back to direct in-process execution. `rbtr index` auto-starts
the daemon unless `--no-daemon` is given.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Annotated

import structlog
from pydantic import BaseModel, BeforeValidator, Field, ValidationError
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
    print_banner,
    print_err,
    print_json_schema,
    progress_reporter,
    render_config,
)
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
    handle_forget,
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
    ForgetRequest,
    ForgetResponse,
    GcMode,
    GcRequest,
    GcResponse,
    ListSymbolsRequest,
    ListSymbolsResponse,
    OkResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    Scope,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
    protocol_json_schema,
)
from rbtr.daemon.server import DaemonServer
from rbtr.daemon.status import DaemonStatusReport, uptime_seconds as _uptime_seconds
from rbtr.errors import RbtrError
from rbtr.git import HEAD_REF, normalise_repo_path, resolve_ref
from rbtr.index.embeddings import Embedder
from rbtr.index.orchestrator import build_index, embed_index
from rbtr.index.reranker import Reranker
from rbtr.index.store import IndexStore
from rbtr.logging import configure_logging

log = structlog.get_logger(__name__)


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
        # The daemon's sole log sink is the rotating JSON file; the
        # parent no longer redirects the child's stderr here.
        configure_logging(to_file=True)

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
        except RbtrError as exc:
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
        except RbtrError as exc:
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
    """Watch refs for continuous indexing (`--remove` to stop).

    Each positional ref is an independent watch target the daemon
    keeps indexed; `rbtr index` with no args watches `HEAD`.
    """

    refs: CliPositionalArg[list[str]] = Field([HEAD_REF], description="Refs to watch and index")
    repo_path: str = Field(".", description="Repository path")
    remove: bool = Field(
        False,
        description="Stop watching the given refs; with no refs, forget a HEAD-only repo",
    )
    remove_stale_refs: bool = Field(
        False,
        description="Stop watching this repo's refs that no longer resolve (deleted branches)",
    )
    remove_stale_repos: bool = Field(
        False,
        description="Forget every indexed repo whose path no longer exists (removed checkouts)",
    )
    daemon: bool = Field(True, description="Use the daemon (disable with --no-daemon)")
    embed: bool = Field(True, description="Compute embeddings (disable with --no-embed)")

    def cli_cmd(self) -> None:
        # Forgetting vanished repos needs no current repo — handle it before
        # resolving the cwd, which would fail outside a git repo.
        if self.remove_stale_repos:
            self._run_remove_stale_repos()
            return

        resolved_repo = normalise_repo_path(self.repo_path)

        if self.remove_stale_refs:
            self._run_remove_stale_refs(resolved_repo)
            return

        if self.remove:
            self._run_remove(resolved_repo)
            return

        # Validate every ref (raises on a bad ref); the daemon stores
        # symbolic names so moving refs track their tip.
        for r in self.refs:
            resolve_ref(resolved_repo, r)

        request = BuildIndexRequest(repo_path=resolved_repo, refs=self.refs, embed=self.embed)

        if not self.daemon:
            self._run_inline(resolved_repo, [resolve_ref(resolved_repo, r) for r in self.refs])
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
            print_err("[dim]Falling back to inline execution.[/]")
            self._run_inline(resolved_repo, [resolve_ref(resolved_repo, r) for r in self.refs])
            return

        self._report_watch(try_daemon(request), started=True)

    def _report_watch(self, resp: object, *, started: bool) -> None:
        """Print the outcome of a daemon watch (add) request."""
        suffix = " (daemon started)" if started else ""
        match resp:
            case BuildIndexResponse():
                emit(resp)
            case OkResponse():
                print_err(f"[green]Watching:[/] {', '.join(self.refs)}{suffix}")
                print_err("[dim]Indexing in background; run `rbtr status` to track.[/]")
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case _:
                print_err("[red]error:[/] daemon did not record the watch")
                sys.exit(1)

    def _run_remove(self, resolved_repo: str) -> None:
        """Stop watching the given refs (daemon path, else inline DB edit).

        Removing `HEAD` forgets the whole repo, but only when `HEAD` is the
        sole watched ref; otherwise it stays rejected (trim the other refs
        first).
        """
        if HEAD_REF in self.refs:
            if not self._current_watched(resolved_repo) - {HEAD_REF}:
                self._run_forget(ForgetRequest(repo_path=resolved_repo))
                return
            print_err("[red]error:[/] HEAD cannot be removed while other refs are watched")
            sys.exit(2)
        request = BuildIndexRequest(
            repo_path=resolved_repo, refs=self.refs, embed=self.embed, remove=True
        )
        resp = try_daemon(request)
        if resp is not None:
            match resp:
                case OkResponse():
                    print_err(f"[green]Stopped watching:[/] {', '.join(self.refs)}")
                case ErrorResponse(message=msg):
                    print_err(f"[red]error:[/] {msg}")
                    sys.exit(1)
                case _:
                    print_err(f"[red]error:[/] unexpected response: {resp}")
                    sys.exit(1)
            return
        # Daemon not running: edit the watch set inline (don't auto-start it).
        store = IndexStore.from_config(writable=True)
        try:
            repo_id = store.get_repo_id(resolved_repo)
            if repo_id is not None:
                with store.session() as ws:
                    ws.remove_watched_refs(repo_id, self.refs)
        finally:
            store.close()
        print_err(f"[green]Stopped watching:[/] {', '.join(self.refs)}")

    def _run_remove_stale_refs(self, resolved_repo: str) -> None:
        """Remove watched refs that no longer resolve (e.g. deleted branches).

        Reuses `status` (which marks unresolvable refs with `sha=None`) when
        the daemon is up; otherwise resolves against the store inline. `HEAD`
        always resolves, so it is never pruned.
        """
        status = try_daemon(StatusRequest(repo_path=resolved_repo)) if self.daemon else None
        if status is not None:
            if not isinstance(status, StatusResponse):
                print_err(f"[red]error:[/] unexpected response: {status}")
                sys.exit(1)
            stale = [w.ref for w in status.watched if w.sha is None and w.ref != HEAD_REF]
            if stale:
                resp = try_daemon(
                    BuildIndexRequest(repo_path=resolved_repo, refs=stale, remove=True)
                )
                if isinstance(resp, ErrorResponse):
                    print_err(f"[red]error:[/] {resp.message}")
                    sys.exit(1)
            self._report_stale(stale)
            return
        # No daemon: resolve each watched ref against the store and prune.
        store = IndexStore.from_config(writable=True)
        try:
            repo_id = store.get_repo_id(resolved_repo)
            stale = []
            if repo_id is not None:
                for ref in store.list_watched_refs(repo_id):
                    if ref == HEAD_REF:
                        continue
                    try:
                        resolve_ref(resolved_repo, ref)
                    except RbtrError:
                        stale.append(ref)
                if stale:
                    with store.session() as ws:
                        ws.remove_watched_refs(repo_id, stale)
        finally:
            store.close()
        self._report_stale(stale)

    @staticmethod
    def _report_stale(stale: list[str]) -> None:
        if stale:
            print_err(f"[green]Removed stale:[/] {', '.join(stale)}")
        else:
            print_err("[dim]No stale watched refs.[/]")

    def _current_watched(self, resolved_repo: str) -> set[str]:
        """The repo's current watch set (daemon `status`, else the store)."""
        status = try_daemon(StatusRequest(repo_path=resolved_repo)) if self.daemon else None
        if isinstance(status, StatusResponse):
            return {w.ref for w in status.watched}
        store = IndexStore.from_config(writable=False)
        try:
            repo_id = store.get_repo_id(resolved_repo)
            return set(store.list_watched_refs(repo_id)) if repo_id is not None else set()
        finally:
            store.close()

    def _run_remove_stale_repos(self) -> None:
        """Forget every indexed repo whose path no longer exists.

        Global and needs no current repo (a vanished path cannot be
        resolved). Daemon path, else inline via the same handler.
        """
        self._run_forget(ForgetRequest(stale=True))

    def _run_forget(self, request: ForgetRequest) -> None:
        """Send a forget request (daemon path, else inline) and report it."""
        resp = try_daemon(request) if self.daemon else None
        if resp is None:
            store = IndexStore.from_config(writable=True)
            try:
                resp = handle_forget(request, store)
            finally:
                store.close()
        match resp:
            case ForgetResponse():
                self._report_forgotten(resp)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case _:
                print_err(f"[red]error:[/] unexpected response: {resp}")
                sys.exit(1)

    @staticmethod
    def _report_forgotten(resp: ForgetResponse) -> None:
        if not resp.forgotten:
            print_err("[dim]Nothing to forget.[/]")
            return
        verb = "Would forget" if resp.dry_run else "Forgot"
        for path in resp.forgotten:
            print_err(f"[green]{verb}:[/] {path}")
        if not resp.dry_run:
            print_err("[dim]Run `rbtr gc` to reclaim the freed space.[/]")

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

            # Each ref is an independent full build.
            result = build_index(
                resolved_repo,
                resolved_refs[0],
                store,
                repo_id=repo_id,
                on_progress=on_progress,
            )
            for ref in resolved_refs[1:]:
                result = build_index(
                    resolved_repo,
                    ref,
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

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)
        weights = None
        if self.alpha is not None and self.beta is not None and self.gamma is not None:
            weights = WeightTriple(alpha=self.alpha, beta=self.beta, gamma=self.gamma)
        try:
            request = SearchRequest(
                repo_path=resolved_repo,
                query=self.query,
                limit=self.limit,
                ref=self.ref,
                weights=weights,
                query_kind=self.query_kind,
                keywords=self.keywords,
                variants=self.variants,
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
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                try:
                    emit(handle_read_symbol(request, store))
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
    repo_path: str = Field(".", description="Repository path")

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)
        request = ListSymbolsRequest(
            repo_path=resolved_repo,
            file_path=self.file,
            ref=self.ref,
        )

        match try_daemon(request):
            case ListSymbolsResponse() as resp:
                emit(resp)
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                try:
                    emit(handle_list_symbols(request, store))
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
            case ErrorResponse(message=msg):
                print_err(f"[red]error:[/] {msg}")
                sys.exit(1)
            case None:
                store = IndexStore.from_config()

                try:
                    emit(handle_find_refs(request, store))
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

    repo_path: str = Field(".", description="Repository path")
    scope: ScopeField = Field(
        Scope.WORKSPACE,
        description="Status scope: workspace (this repo) or all (every indexed repo).",
    )

    def cli_cmd(self) -> None:
        resolved_repo = normalise_repo_path(self.repo_path)
        request = StatusRequest(repo_path=resolved_repo, scope=self.scope)

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

    Destructive and not undoable — permanently deletes indexed
    commits/chunks. Use --dry-run to preview first.

    Operates on the current repo by default; `--all-repos` reclaims
    across every indexed repo at once, but only with the safe default
    (watched) reclamation — scope aggressive modes to one repo.

    By default keeps the watch set — HEAD, all local branches/tags,
    and every watched ref (plus the current worktree) — and sweeps
    crashed-build residue. Alternate modes let you keep only the
    watch set, only HEAD, or specific refs, or sweep residue only.
    """

    repo_path: str = Field(".", description="Repository path")
    all_repos: bool = Field(
        False,
        description="GC every indexed repo (default reclamation only; not with aggressive modes)",
    )
    watched_only: bool = Field(
        False,
        description="Keep only HEAD and watched refs (drop unwatched branches/tags)",
    )
    keep_head_only: bool = Field(False, description="Keep only HEAD (drop everything else)")
    keep: CliPositionalArg[list[str]] = Field(
        [],
        description="Keep only these refs plus HEAD; drop the rest",
    )
    orphans: bool = Field(
        False,
        description="Sweep crashed-build residue only; no commits dropped",
    )
    dry_run: bool = Field(False, description="Report what would be removed without writing")

    def cli_cmd(self) -> None:
        mode, refs = self._resolve_mode()
        if self.all_repos:
            if mode is not GcMode.WATCHED:
                print_err(
                    "[red]error:[/] --all-repos supports only the default reclamation; "
                    "scope an aggressive mode with --repo-path"
                )
                sys.exit(2)
            resolved_repo = None
        else:
            resolved_repo = normalise_repo_path(self.repo_path)
        request = GcRequest(
            repo_path=resolved_repo,
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
        """Pick the GC mode from the set of flags."""
        if self.orphans:
            return GcMode.ORPHANS, []
        if self.keep:
            return GcMode.KEEP, self.keep
        if self.watched_only:
            return GcMode.WATCHED_ONLY, []
        if self.keep_head_only:
            return GcMode.HEAD_ONLY, []
        # Default: HEAD + local branches/tags + the watch set.
        return GcMode.WATCHED, []


class SchemaDump(BaseModel):
    """Dump JSON Schema for the daemon protocol.

    Useful for generating TypeScript types or validating messages.
    """

    def cli_cmd(self) -> None:
        # Always outputs JSON — ignore json_output mode.
        print_json_schema(protocol_json_schema())


class ConfigCmd(BaseModel):
    """Show the rendered rbtr configuration."""

    def cli_cmd(self) -> None:
        render_config()


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
    at the outer boundary so subcommand bodies don't each have to
    do the same try/except dance.
    """
    configure_logging()
    cli_source: CliSettingsSource[Rbtr] = CliSettingsSource(Rbtr, formatter_class=RichHelpFormatter)
    try:
        CliApp.run(Rbtr, cli_settings_source=cli_source)
    except RbtrError as exc:
        print_err(f"[red]error:[/] {exc}")
        sys.exit(2)
