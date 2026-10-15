"""Request handlers for the daemon server.

Each handler takes a typed request and returns the success
response directly. Errors are signalled by raising `RbtrError`
(or subclasses like `IndexNotBuiltError`).

The daemon's `_dispatch` wraps every handler call in
`try/except Exception` and converts unhandled errors to
`ErrorResponse` for the protocol. CLI callers let exceptions
propagate to `main()`, which prints them.

Every handler takes the index as `store` (an `IndexStore`); the
writing ones (`gc`, `forget`, `index`) open a session on it, which
the daemon runs in a thread under its write lock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from rbtr import get_version
from rbtr.config import config
from rbtr.daemon.dto import (
    PluginInfo,
    RefOuts,
    ResolvedRef,
    SearchHitOut,
    SymbolOut,
    SymbolRefOut,
)
from rbtr.daemon.messages import (
    ActiveJob,
    ChangedSymbol,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    DaemonConfigRequest,
    DaemonConfigResponse,
    FindRefsRequest,
    FindRefsResponse,
    ForgetRequest,
    ForgetResponse,
    ForgetStaleRequest,
    GcAllRequest,
    GcRequest,
    GcResponse,
    IndexedRef,
    ListSymbolsRequest,
    ListSymbolsResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    SearchAllRequest,
    SearchQuery,
    SearchRequest,
    SearchResponse,
    StatusAllRequest,
    StatusRequest,
    StatusResponse,
    UnwatchRequest,
    UnwatchResponse,
    UnwatchStaleAllRequest,
    UnwatchStaleRequest,
    WatchedRef,
    WatchRequest,
    WatchSetResponse,
)
from rbtr.domain.models import (
    Chunk,
    GcCounts,
    QueryKind,
    RefSource,
    Repo,
    SnapshotCounts,
    SnapshotRange,
    SnapshotRef,
)
from rbtr.errors import IndexNotBuiltError, MissingSnapshot, RbtrError
from rbtr.git import (
    HEAD_REF,
    WORKTREE_REF,
    names_for_commits,
    resolve_ref,
    untracked_file_count,
)
from rbtr.index.gc import run_gc, run_gc_all
from rbtr.index.progress import RepoProgressCallback, _noop_repo_progress
from rbtr.index.results import changed_to_symbols
from rbtr.index.search import search
from rbtr.index.watch import forget_stale_repos, remove_stale_refs, unwatch_refs
from rbtr.languages.manager import get_manager

if TYPE_CHECKING:
    from rbtr.index.embeddings import Embedder
    from rbtr.index.reranker import Reranker
    from rbtr.index.store import IndexStore

log = structlog.get_logger(__name__)


def resolve_refs(repo_path: str, refs: list[str]) -> list[str]:
    """Resolve symbolic refs to commit SHAs (build path).

    Raises `RbtrError` if any ref cannot be resolved.
    """
    return [
        resolve_ref(repo_path, ref, max_untracked_files=config.max_untracked_files) for ref in refs
    ]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedRead:
    """The snapshot a read resolved to, and how it was chosen."""

    at: SnapshotRef
    source: RefSource


def _resolve_read_ref(
    store: IndexStore,
    repo_path: str,
    repo_id: int,
    requested_ref: str | None,
    *,
    require_indexed: bool = False,
) -> ResolvedRead:
    """Resolve a ref for read operations to the snapshot it names.

    When *requested_ref* is `None`, prefers the worktree's tree SHA
    when the tree is dirty and indexed (`indexed_worktree_ref`).
    Otherwise resolves `"HEAD"`.  Explicit `"HEAD"` always resolves
    to the committed state.  Falls back to the latest indexed commit
    when the repo is missing.  Raises `RbtrError` if the ref cannot
    be resolved.

    When *require_indexed* is set, the resolved SHA must be usable:

    - An explicit *requested_ref* whose SHA is not indexed is an
      error (`_require_indexed`) rather than a silent empty result.
    - The implicit worktree/HEAD path falls back to the latest
      indexed commit when the resolved SHA is not indexed (e.g. a
      build is still finalising), and only errors when the repo has
      no indexed commits at all. So an older indexed version of a
      symbol is preferred over an error.

    The result says which of these choices was made (`RefSource`).
    """

    explicit = requested_ref is not None
    if requested_ref is None:
        if (dirty := store.indexed_worktree_ref(repo_path, repo_id)) is not None:
            return ResolvedRead(at=dirty, source=RefSource.WORKTREE)
        requested_ref = HEAD_REF
    try:
        sha = resolve_ref(repo_path, requested_ref, max_untracked_files=config.max_untracked_files)
    except RbtrError:
        if requested_ref == HEAD_REF and (latest := store.latest_indexed_ref(repo_id)) is not None:
            return ResolvedRead(at=latest, source=RefSource.LATEST_INDEXED)
        msg = f"Cannot resolve ref '{requested_ref}' in {repo_path}"
        raise RbtrError(msg) from None
    at = SnapshotRef(repo_id=repo_id, snapshot_sha=sha)
    if require_indexed and not store.has_indexed(at=at):
        if not explicit and (latest := store.latest_indexed_ref(repo_id)) is not None:
            return ResolvedRead(at=latest, source=RefSource.LATEST_INDEXED)
        _require_indexed(store, at, requested_ref)
    return ResolvedRead(at=at, source=RefSource.REQUESTED if explicit else RefSource.HEAD)


# ── Read-only handlers ───────────────────────────────────────────────


def handle_search(
    request: SearchRequest,
    store: IndexStore,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> SearchResponse:
    """Search the repo at `request.repo_path` for `request.query`.

    Propagates `IndexNotBuiltError` from the store; the daemon's
    `_dispatch` turns it into an "index is building" message when a
    build is active.
    """
    repo_id = store.resolve_repo(request.repo_path)
    read = _resolve_read_ref(store, request.repo_path, repo_id, request.ref, require_indexed=True)
    return _search(
        request,
        store,
        [read.at],
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        repo_paths=None,
        embedder=embedder,
        reranker=reranker,
    )


def handle_search_all(
    request: SearchAllRequest,
    store: IndexStore,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> SearchResponse:
    """Search every indexed repo, attributing each result with its `repo_path`."""
    return _search(
        request,
        store,
        store.list_latest_refs(),
        resolved=None,
        repo_paths={r.repo_id: r.repo_path for r in store.list_repos()},
        embedder=embedder,
        reranker=reranker,
    )


def _search(
    request: SearchQuery,
    store: IndexStore,
    within: list[SnapshotRef],
    *,
    resolved: ResolvedRef | None,
    repo_paths: dict[int, str] | None,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> SearchResponse:
    """Run *request* over the snapshots *within*."""
    override = QueryKind(request.query_kind) if request.query_kind else None
    results = search(
        store,
        request.query,
        within=within,
        top_k=request.limit,
        embedder=embedder,
        kind=override,
        keywords=request.keywords,
        variants=request.variants,
        weights=request.weights,
        reranker=reranker,
        reranker_pool=request.reranker_pool,
        reranker_blend_weight=request.reranker_blend_weight,
        repo_paths=repo_paths,
    )
    query_kind = override or (results[0].query_kind if results else None)
    return SearchResponse(
        results=[SearchHitOut.from_scored(r, explain=request.explain) for r in results],
        resolved=resolved,
        query_kind=query_kind if request.explain else None,
    )


def _scope_chunks(chunks: list[Chunk], file_paths: list[str] | None) -> list[Chunk]:
    """Filter chunks to *file_paths*; a no-op when it is empty or `None`."""
    if not file_paths:
        return chunks
    allowed = set(file_paths)
    return [c for c in chunks if c.file_path in allowed]


def handle_read_symbol(request: ReadSymbolRequest, store: IndexStore) -> ReadSymbolResponse:
    repo_id = store.resolve_repo(request.repo_path)
    read = _resolve_read_ref(store, request.repo_path, repo_id, request.ref, require_indexed=True)
    scoped = _scope_chunks(store.match_by_name(request.symbol, at=read.at), request.file_paths)
    return ReadSymbolResponse(
        chunks=[SymbolOut.from_chunk(c) for c in scoped],
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        file_paths=request.file_paths,
    )


def handle_list_symbols(request: ListSymbolsRequest, store: IndexStore) -> ListSymbolsResponse:
    repo_id = store.resolve_repo(request.repo_path)
    read = _resolve_read_ref(store, request.repo_path, repo_id, request.ref, require_indexed=True)
    chunks = store.get_chunks(at=read.at, file_path=request.file_path)
    return ListSymbolsResponse(
        chunks=[SymbolRefOut.from_chunk(c) for c in chunks],
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        file_path=request.file_path,
    )


def handle_find_refs(request: FindRefsRequest, store: IndexStore) -> FindRefsResponse:
    repo_id = store.resolve_repo(request.repo_path)
    read = _resolve_read_ref(store, request.repo_path, repo_id, request.ref, require_indexed=True)
    chunks = _scope_chunks(store.match_by_name(request.symbol, at=read.at), request.file_paths)
    frame = store.inbound_refs([chunk.id for chunk in chunks], at=read.at)
    refs = RefOuts.validate_python(frame.to_dicts())
    return FindRefsResponse(
        refs=refs,
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        file_paths=request.file_paths,
    )


def _require_indexed(store: IndexStore, at: SnapshotRef, requested_ref: str) -> None:
    """Raise `IndexNotBuiltError` if the snapshot at *at* is not indexed.

    *requested_ref* is what the client asked for, which the message
    quotes; *at* is what it resolved to.

    The daemon's `_dispatch` upgrades this to an "index is building"
    message when it will index that snapshot; inline callers see the
    plain "not indexed" guidance.
    """
    if store.has_indexed(at=at):
        return
    if requested_ref == WORKTREE_REF:
        msg = "Working tree is not indexed yet — run rbtr watch first"
    else:
        msg = f"Ref '{requested_ref}' is not indexed — run rbtr watch first"
    missing = MissingSnapshot(repo_id=at.repo_id, snapshot_sha=at.snapshot_sha, ref=requested_ref)
    raise IndexNotBuiltError(msg, missing=missing)


def handle_changed_symbols(
    request: ChangedSymbolsRequest, store: IndexStore
) -> ChangedSymbolsResponse:
    """Diff two indexed refs at the symbol level.

    Both refs must already be indexed; an unindexed side is an
    error rather than an empty diff (the symbol-level comparison
    reads chunks from both commits).
    """
    repo_id = store.resolve_repo(request.repo_path)
    base = resolve_ref(
        request.repo_path, request.base, max_untracked_files=config.max_untracked_files
    )
    head = resolve_ref(
        request.repo_path, request.head, max_untracked_files=config.max_untracked_files
    )
    _require_indexed(store, SnapshotRef(repo_id=repo_id, snapshot_sha=base), request.base)
    _require_indexed(store, SnapshotRef(repo_id=repo_id, snapshot_sha=head), request.head)
    frame = store.changed_symbols(
        between=SnapshotRange(repo_id=repo_id, base_sha=base, head_sha=head),
        file_paths=request.file_paths,
    )
    changes = [
        ChangedSymbol(chunk=SymbolRefOut.from_chunk(chunk), change=change)
        for chunk, change in changed_to_symbols(frame)
    ]
    return ChangedSymbolsResponse(
        changes=changes, base_sha=base, head_sha=head, file_paths=request.file_paths
    )


# The active build and embed of the repo at a path, or of any repo for `None`.
type SnapshotStatusFn = Callable[[str | None], tuple[ActiveJob | None, ActiveJob | None]]


def _indexed_refs(
    repos: Sequence[Repo],
    counted: Sequence[tuple[SnapshotRef, SnapshotCounts]],
) -> list[IndexedRef]:
    """Build one `IndexedRef` per counted snapshot, in the order given.

    *counted* already carries every indexed snapshot, its two figures and
    its display order, so the work left is resolving each repo's symbolic
    ref names — one git call per repo.
    """
    paths = {repo.repo_id: repo.repo_path for repo in repos}
    shas: dict[int, list[str]] = {}
    for ref, _ in counted:
        shas.setdefault(ref.repo_id, []).append(ref.snapshot_sha)
    names = {
        repo_id: names_for_commits(paths[repo_id], of_repo) for repo_id, of_repo in shas.items()
    }
    return [
        IndexedRef(
            sha=ref.snapshot_sha,
            names=names[ref.repo_id].get(ref.snapshot_sha, []),
            total=count.total,
            embedded=count.embedded,
            repo_path=paths[ref.repo_id],
        )
        for ref, count in counted
    ]


def _watched_for_repo(
    store: IndexStore,
    repo_id: int,
    repo_path: str,
) -> list[WatchedRef]:
    """Build `WatchedRef`s for one repo's watch set.

    Resolves each watched ref to a SHA (`None` if it no longer
    resolves) and marks it indexed when that SHA is recorded in
    `indexed_snapshots`; otherwise it is pending.
    """
    out: list[WatchedRef] = []
    for ref in store.list_watched_refs(repo_id):
        try:
            sha: str | None = resolve_ref(
                repo_path, ref, max_untracked_files=config.max_untracked_files
            )
        except RbtrError:
            sha = None
        out.append(
            WatchedRef(
                ref=ref,
                sha=sha,
                indexed=sha is not None
                and store.has_indexed(at=SnapshotRef(repo_id=repo_id, snapshot_sha=sha)),
                repo_path=repo_path,
            )
        )
    return out


def handle_status(
    request: StatusRequest,
    store: IndexStore,
    snapshot_status: SnapshotStatusFn | None = None,
) -> StatusResponse:
    """Report index status for the repo at `request.repo_path`."""
    repo_id = store.get_repo_id(request.repo_path)
    if repo_id is None:
        return StatusResponse(
            repo_path=request.repo_path,
            db_path=store.db_path,
            db_size_bytes=store.disk_size_bytes(),
            indexed_refs=[],
            watched=[],
            active_build=None,
            active_embed=None,
        )
    return _status(
        store,
        request.repo_path,
        [Repo(repo_id=repo_id, repo_path=request.repo_path)],
        store.chunk_counts_by_snapshot(repo_id=repo_id),
        snapshot_status(request.repo_path) if snapshot_status is not None else (None, None),
    )


def handle_status_all(
    _request: StatusAllRequest,
    store: IndexStore,
    snapshot_status: SnapshotStatusFn | None = None,
) -> StatusResponse:
    """Report index status for every indexed repo."""
    # Counts before repos, so every repo the counts name has a path
    # here to render it, even one registered between the two calls.
    counted = store.chunk_counts_by_snapshot()
    return _status(
        store,
        None,
        store.list_repos(),
        counted,
        snapshot_status(None) if snapshot_status is not None else (None, None),
    )


def _status(
    store: IndexStore,
    repo_path: str | None,
    repos: list[Repo],
    counted: Sequence[tuple[SnapshotRef, SnapshotCounts]],
    active: tuple[ActiveJob | None, ActiveJob | None],
) -> StatusResponse:
    """The status of *repos*, given their *counted* snapshots and the *active* jobs."""
    return StatusResponse(
        repo_path=repo_path,
        db_path=store.db_path,
        db_size_bytes=store.disk_size_bytes(),
        indexed_refs=_indexed_refs(repos, counted),
        watched=[
            watched
            for repo in repos
            for watched in _watched_for_repo(store, repo.repo_id, repo.repo_path)
        ],
        untracked_not_indexed={
            repo.repo_path: count
            for repo in repos
            if (count := untracked_file_count(repo.repo_path)) > config.max_untracked_files
        },
        active_build=active[0],
        active_embed=active[1],
    )


def handle_daemon_config(_request: DaemonConfigRequest) -> DaemonConfigResponse:
    """Report the daemon's live config and the language plugins it loaded.

    Independent of any repo or index — answers from the daemon process's
    own `config` and `LanguageManager`. Each plugin joins its registration
    (for `extraction_serial`) with its distribution (for package/version).
    """
    mgr = get_manager()
    plugins: list[PluginInfo] = []
    for language in sorted(mgr.all_language_ids()):
        dist = mgr.distribution(language)
        reg = mgr.get_registration(language)
        if dist is None or reg is None:
            continue
        package, version = dist
        plugins.append(
            PluginInfo(
                language=language,
                package=package,
                version=version,
                extraction_serial=reg.extraction_serial,
            )
        )
    return DaemonConfigResponse(
        rbtr_version=get_version(),
        config=config.model_dump(mode="json"),
        plugins=plugins,
    )


# ── Build handler ────────────────────────────────────────────────────


def handle_gc(
    request: GcRequest,
    store: IndexStore,
    *,
    allow_compact: bool = False,
) -> GcResponse:
    """Reclaim index storage in the repo at `request.repo_path`."""
    t0 = time.monotonic()
    size_before = store.disk_size_bytes()
    # Compaction rewrites and swaps the database file, so it is only safe
    # when the caller owns the connection exclusively -- the inline
    # no-daemon path (`allow_compact=True`). The daemon shares the
    # connection with live searches and leaves it off.
    counts = run_gc(
        store,
        request.repo_path,
        mode=request.mode,
        refs=request.refs,
        dry_run=request.dry_run,
        compact=request.compact and allow_compact and not request.dry_run,
    )
    return _gc_response(request, store, counts, repos_collected=1, t0=t0, size_before=size_before)


def handle_gc_all(
    request: GcAllRequest,
    store: IndexStore,
    *,
    allow_compact: bool = False,
    on_progress: RepoProgressCallback = _noop_repo_progress,
) -> GcResponse:
    """Reclaim index storage across every indexed repo, reporting each repo collected."""
    t0 = time.monotonic()
    size_before = store.disk_size_bytes()
    counts, repos_collected = run_gc_all(
        store,
        mode=request.mode,
        dry_run=request.dry_run,
        compact=request.compact and allow_compact and not request.dry_run,  # as in `handle_gc`
        on_progress=on_progress,
    )
    return _gc_response(
        request, store, counts, repos_collected=repos_collected, t0=t0, size_before=size_before
    )


def _gc_response(
    request: GcRequest | GcAllRequest,
    store: IndexStore,
    counts: GcCounts,
    *,
    repos_collected: int,
    t0: float,
    size_before: int,
) -> GcResponse:
    """Report a gc that collected *counts* from *repos_collected* repos since *t0*."""
    # A real run's `counts.chunks` is the actual reclamation. A dry run only
    # predicts the drop set's freed chunks, so add the pre-existing orphans
    # the global prune would also remove (counted once across all repos).
    chunks_freed = counts.chunks
    if request.dry_run:
        chunks_freed += store.count_orphan_chunks()
    size_after = store.disk_size_bytes()
    elapsed = time.monotonic() - t0
    log.info(
        "gc_complete",
        mode=request.mode,
        dry_run=request.dry_run,
        repos=repos_collected,
        snapshots=counts.snapshots,
        file_snapshots=counts.file_snapshots,
        edges=counts.edges,
        chunks_freed=chunks_freed,
        elapsed_ms=round(elapsed * 1000, 1),
    )
    return GcResponse(
        repos_collected=repos_collected,
        snapshots_dropped=counts.snapshots,
        file_snapshots_dropped=counts.file_snapshots,
        edges_dropped=counts.edges,
        chunks_freed=chunks_freed,
        size_before_bytes=size_before,
        size_after_bytes=size_after,
        elapsed_seconds=elapsed,
        dry_run=request.dry_run,
    )


def handle_unwatch(request: UnwatchRequest, store: IndexStore) -> UnwatchResponse:
    """Drop the refs the request names."""
    removed = unwatch_refs(
        store, repo_path=request.repo_path, refs=request.refs, dry_run=request.dry_run
    )
    log.info(
        "watched_refs_removed",
        repo=request.repo_path,
        refs=request.refs,
        dry_run=request.dry_run,
    )
    return UnwatchResponse(removed=removed, dry_run=request.dry_run)


def handle_unwatch_stale(request: UnwatchStaleRequest, store: IndexStore) -> UnwatchResponse:
    """Drop the refs git can no longer resolve in the repo at `request.repo_path`.

    Raises `RbtrError` if that repo was never indexed.
    """
    repo = Repo(repo_id=store.resolve_repo(request.repo_path), repo_path=request.repo_path)
    return _unwatch_stale([repo], store, dry_run=request.dry_run)


def handle_unwatch_stale_all(request: UnwatchStaleAllRequest, store: IndexStore) -> UnwatchResponse:
    """Drop the refs git can no longer resolve in every indexed repo."""
    return _unwatch_stale(store.list_repos(), store, dry_run=request.dry_run)


def _unwatch_stale(repos: list[Repo], store: IndexStore, *, dry_run: bool) -> UnwatchResponse:
    """Drop the refs git can no longer resolve in *repos*."""
    removed = remove_stale_refs(store, repos, dry_run=dry_run)
    log.info(
        "stale_refs_removed",
        dry_run=dry_run,
        repos=len(removed),
        refs=sum(len(refs) for refs in removed.values()),
    )
    return UnwatchResponse(removed=removed, dry_run=dry_run)


def handle_forget_stale(request: ForgetStaleRequest, store: IndexStore) -> ForgetResponse:
    """Forget the repos whose checkout is gone."""
    gone = forget_stale_repos(store, dry_run=request.dry_run)
    return ForgetResponse(forgotten=[repo.repo_path for repo in gone], dry_run=request.dry_run)


def handle_forget(request: ForgetRequest, store: IndexStore) -> ForgetResponse:
    """Forget one repo (metadata-only; GC reclaims the chunks).

    Only when its watch set is exactly `{HEAD}` — trim other refs
    first. `dry_run` reports without deleting.
    """
    target_id = store.get_repo_id(request.repo_path)
    if target_id is None:
        return ForgetResponse(forgotten=[], dry_run=request.dry_run)
    # Forget only when nothing beyond HEAD is watched (an empty watch set —
    # e.g. an inline `--no-daemon` index — also qualifies).
    if set(store.list_watched_refs(target_id)) - {HEAD_REF}:
        msg = "refusing to forget a repo watching refs beyond HEAD; remove them first"
        raise RbtrError(msg)
    if not request.dry_run:
        with store.session() as ws:
            ws.forget_repo(target_id)
    return ForgetResponse(forgotten=[request.repo_path], dry_run=request.dry_run)


def handle_watch(
    request: WatchRequest,
    store: IndexStore,
) -> WatchSetResponse:
    """Record the request's refs in the repo's watch set, and return the set.

    The worker derives and runs the actual build from `watched_refs`
    on its next poll.  `HEAD` is always included, so a repo first seen
    here watches it as one seen at startup does.  Dropping refs is
    `handle_unwatch`.
    """
    with store.session() as ws:
        repo_id = ws.register_repo(request.repo_path)
        ws.add_watched_refs(repo_id, [HEAD_REF, *request.refs])
    log.info("watched_refs_added", repo=request.repo_path, refs=request.refs)
    return WatchSetResponse(watched=_watched_for_repo(store, repo_id, request.repo_path))
