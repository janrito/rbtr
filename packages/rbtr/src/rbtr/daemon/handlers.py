"""Request handlers for the daemon server.

Each handler takes a typed request and returns the success
response directly. Errors are signalled by raising `RbtrError`
(or subclasses like `IndexNotBuiltError`).

The daemon's `_dispatch` wraps every handler call in
`try/except Exception` and converts unhandled errors to
`ErrorResponse` for the protocol. CLI callers let exceptions
propagate to `main()`, which prints them.

Every handler takes the index as `index` (an `Index`); the
writing ones (`gc`, `forget`, `index`) open a session on it, which
the daemon runs in a thread under its write lock.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
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
    IndexState,
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
from rbtr.errors import RbtrError
from rbtr.git import (
    HEAD_REF,
    names_for_commits,
    resolve_ref,
    untracked_file_count,
)
from rbtr.index.gc import run_gc, run_gc_all
from rbtr.index.progress import RepoProgressCallback, _noop_repo_progress
from rbtr.index.refs import ensure_indexed, latest_refs, resolve_read_ref
from rbtr.index.results import changed_to_symbols, frame_to_chunks
from rbtr.index.search import search
from rbtr.index.watch import forget_stale_repos, remove_stale_refs, unwatch_refs
from rbtr.languages.manager import get_manager

if TYPE_CHECKING:
    from rbtr.index.embeddings import Embedder
    from rbtr.index.index import Index
    from rbtr.index.reranker import Reranker

log = structlog.get_logger(__name__)


def handle_search(
    request: SearchRequest,
    index: Index,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> SearchResponse:
    """Search the repo at `request.repo_path` for `request.query`.

    Propagates `IndexNotBuiltError` from the store; the daemon's
    `_dispatch` turns it into an "index is building" message when a
    build is active.
    """
    repo = index.repo(request.repo_path)
    read = resolve_read_ref(index, repo, request.ref, require_indexed=True)
    return _search(
        request,
        index,
        [read.at],
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        attribute_repos=False,
        embedder=embedder,
        reranker=reranker,
    )


def handle_search_all(
    request: SearchAllRequest,
    index: Index,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> SearchResponse:
    """Search every indexed repo, attributing each result with its `repo_path`."""
    return _search(
        request,
        index,
        latest_refs(index),
        resolved=None,
        attribute_repos=True,
        embedder=embedder,
        reranker=reranker,
    )


def _search(
    request: SearchQuery,
    index: Index,
    within: list[SnapshotRef],
    *,
    resolved: ResolvedRef | None,
    attribute_repos: bool,
    embedder: Embedder | None,
    reranker: Reranker | None,
) -> SearchResponse:
    """Run *request* over the snapshots *within*."""
    override = QueryKind(request.query_kind) if request.query_kind else None
    results = search(
        index,
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
        attribute_repos=attribute_repos,
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


def handle_read_symbol(request: ReadSymbolRequest, index: Index) -> ReadSymbolResponse:
    repo = index.repo(request.repo_path)
    read = resolve_read_ref(index, repo, request.ref, require_indexed=True)
    named = frame_to_chunks(index.name_matches(request.symbol, within=[read.at]))
    scoped = _scope_chunks(named, request.file_paths)
    return ReadSymbolResponse(
        chunks=[SymbolOut.from_chunk(c) for c in scoped],
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        file_paths=request.file_paths,
    )


def handle_list_symbols(request: ListSymbolsRequest, index: Index) -> ListSymbolsResponse:
    repo = index.repo(request.repo_path)
    read = resolve_read_ref(index, repo, request.ref, require_indexed=True)
    chunks = index.chunks(at=read.at, file_path=request.file_path)
    return ListSymbolsResponse(
        chunks=[SymbolRefOut.from_chunk(c) for c in chunks],
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        file_path=request.file_path,
    )


def handle_find_refs(request: FindRefsRequest, index: Index) -> FindRefsResponse:
    repo = index.repo(request.repo_path)
    read = resolve_read_ref(index, repo, request.ref, require_indexed=True)
    named = frame_to_chunks(index.name_matches(request.symbol, within=[read.at]))
    chunks = _scope_chunks(named, request.file_paths)
    frame = index.inbound_refs([chunk.id for chunk in chunks], at=read.at)
    refs = RefOuts.validate_python(frame.to_dicts())
    return FindRefsResponse(
        refs=refs,
        resolved=ResolvedRef(sha=read.at.snapshot_sha, source=read.source),
        file_paths=request.file_paths,
    )


def handle_changed_symbols(request: ChangedSymbolsRequest, index: Index) -> ChangedSymbolsResponse:
    """Diff two indexed refs at the symbol level.

    Both refs must already be indexed; an unindexed side is an
    error rather than an empty diff (the symbol-level comparison
    reads chunks from both commits).
    """
    repo = index.repo(request.repo_path)
    base = resolve_ref(
        request.repo_path, request.base, max_untracked_files=config.max_untracked_files
    )
    head = resolve_ref(
        request.repo_path, request.head, max_untracked_files=config.max_untracked_files
    )
    ensure_indexed(index, repo.at(base), request.base)
    ensure_indexed(index, repo.at(head), request.head)
    frame = index.changed_symbols(
        between=SnapshotRange(repo=repo, base_sha=base, head_sha=head),
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


def _indexed_refs(counted: Mapping[SnapshotRef, SnapshotCounts]) -> list[IndexedRef]:
    """Build one `IndexedRef` per counted snapshot, in the order given.

    *counted* already carries every indexed snapshot, its two figures and
    its display order, so the work left is resolving each repo's symbolic
    ref names — one git call per repo.
    """
    shas: dict[Repo, list[str]] = {}
    for ref in counted:
        shas.setdefault(ref.repo, []).append(ref.snapshot_sha)
    names = {repo: names_for_commits(repo.repo_path, of_repo) for repo, of_repo in shas.items()}
    return [
        IndexedRef(
            sha=ref.snapshot_sha,
            names=names[ref.repo].get(ref.snapshot_sha, []),
            total=count.total,
            embedded=count.embedded,
            repo_path=ref.repo.repo_path,
        )
        for ref, count in counted.items()
    ]


def _watched_for_repo(index: Index, repo: Repo) -> list[WatchedRef]:
    """Build `WatchedRef`s for one repo's watch set.

    Resolves each watched ref to a SHA (`None` if it no longer
    resolves) and marks it indexed when that SHA is recorded in
    `indexed_snapshots`; otherwise it is pending.
    """
    out: list[WatchedRef] = []
    for ref in index.watched_refs(repo):
        try:
            sha: str | None = resolve_ref(
                repo.repo_path, ref, max_untracked_files=config.max_untracked_files
            )
        except RbtrError:
            sha = None
        out.append(
            WatchedRef(
                ref=ref,
                sha=sha,
                indexed=sha is not None and index.has_indexed(at=repo.at(sha)),
                repo_path=repo.repo_path,
            )
        )
    return out


def handle_status(
    request: StatusRequest,
    index: Index,
    snapshot_status: SnapshotStatusFn | None = None,
) -> StatusResponse:
    """Report index status for the repo at `request.repo_path`."""
    repo = index.find_repo(request.repo_path)
    if repo is None:
        return StatusResponse(
            state=IndexState.REPO_NOT_INDEXED if index.repos() else IndexState.NO_INDEX,
            repo_path=request.repo_path,
            db_path=index.db_path,
            db_size_bytes=index.disk_size_bytes(),
            indexed_refs=[],
            watched=[],
            active_build=None,
            active_embed=None,
        )
    return _status(
        index,
        _read_state(index, repo),
        request.repo_path,
        [repo],
        index.chunk_counts_by_snapshot(repo=repo),
        snapshot_status(request.repo_path) if snapshot_status is not None else (None, None),
    )


def handle_status_all(
    _request: StatusAllRequest,
    index: Index,
    snapshot_status: SnapshotStatusFn | None = None,
) -> StatusResponse:
    """Report index status for every indexed repo."""
    repos = index.repos()
    counted = index.chunk_counts_by_snapshot()
    if not repos:
        state = IndexState.NO_INDEX
    elif counted:
        state = IndexState.INDEXED
    else:
        state = IndexState.REPO_NOT_INDEXED
    return _status(
        index,
        state,
        None,
        repos,
        counted,
        snapshot_status(None) if snapshot_status is not None else (None, None),
    )


def _read_state(index: Index, repo: Repo) -> IndexState:
    """Whether a read of *repo* without a ref finds the snapshot it asks for (`resolve_read_ref`)."""
    try:
        read = resolve_read_ref(index, repo, None, require_indexed=True)
    except RbtrError:
        return IndexState.REPO_NOT_INDEXED
    if read.source is RefSource.LATEST_INDEXED:
        return IndexState.REF_NOT_INDEXED
    return IndexState.INDEXED


def _status(
    index: Index,
    state: IndexState,
    repo_path: str | None,
    repos: list[Repo],
    counted: Mapping[SnapshotRef, SnapshotCounts],
    active: tuple[ActiveJob | None, ActiveJob | None],
) -> StatusResponse:
    """The status of *repos*, given their *counted* snapshots and the *active* jobs."""
    return StatusResponse(
        state=state,
        repo_path=repo_path,
        db_path=index.db_path,
        db_size_bytes=index.disk_size_bytes(),
        indexed_refs=_indexed_refs(counted),
        watched=[watched for repo in repos for watched in _watched_for_repo(index, repo)],
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


def handle_gc(
    request: GcRequest,
    index: Index,
    *,
    allow_compact: bool = False,
) -> GcResponse:
    """Reclaim index storage in the repo at `request.repo_path`."""
    t0 = time.monotonic()
    size_before = index.disk_size_bytes()
    # Compaction switches the repo's project to a rewritten file, which a
    # write committing to the old file meanwhile would miss; only a caller
    # that holds the writes passes `allow_compact=True`: the inline
    # no-daemon path, and the daemon under `_write_sem`.
    counts = run_gc(
        index,
        request.repo_path,
        mode=request.mode,
        refs=request.refs,
        dry_run=request.dry_run,
        compact=request.compact and allow_compact and not request.dry_run,
    )
    return _gc_response(request, index, counts, repos_collected=1, t0=t0, size_before=size_before)


def handle_gc_all(
    request: GcAllRequest,
    index: Index,
    *,
    allow_compact: bool = False,
    on_progress: RepoProgressCallback = _noop_repo_progress,
) -> GcResponse:
    """Reclaim index storage across every indexed repo, reporting each repo collected."""
    t0 = time.monotonic()
    size_before = index.disk_size_bytes()
    counts, repos_collected = run_gc_all(
        index,
        mode=request.mode,
        dry_run=request.dry_run,
        compact=request.compact and allow_compact and not request.dry_run,  # as in `handle_gc`
        on_progress=on_progress,
    )
    return _gc_response(
        request, index, counts, repos_collected=repos_collected, t0=t0, size_before=size_before
    )


def _gc_response(
    request: GcRequest | GcAllRequest,
    index: Index,
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
        chunks_freed += index.count_orphan_chunks()
    size_after = index.disk_size_bytes()
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


def handle_unwatch(request: UnwatchRequest, index: Index) -> UnwatchResponse:
    """Drop the refs the request names."""
    removed = unwatch_refs(
        index, repo_path=request.repo_path, refs=request.refs, dry_run=request.dry_run
    )
    log.info(
        "watched_refs_removed",
        repo=request.repo_path,
        refs=request.refs,
        dry_run=request.dry_run,
    )
    return UnwatchResponse(removed=removed, dry_run=request.dry_run)


def handle_unwatch_stale(request: UnwatchStaleRequest, index: Index) -> UnwatchResponse:
    """Drop the refs git can no longer resolve in the repo at `request.repo_path`.

    Raises `RbtrError` if that repo was never indexed.
    """
    return _unwatch_stale([index.repo(request.repo_path)], index, dry_run=request.dry_run)


def handle_unwatch_stale_all(request: UnwatchStaleAllRequest, index: Index) -> UnwatchResponse:
    """Drop the refs git can no longer resolve in every indexed repo."""
    return _unwatch_stale(index.repos(), index, dry_run=request.dry_run)


def _unwatch_stale(repos: list[Repo], index: Index, *, dry_run: bool) -> UnwatchResponse:
    """Drop the refs git can no longer resolve in *repos*."""
    removed = remove_stale_refs(index, repos, dry_run=dry_run)
    log.info(
        "stale_refs_removed",
        dry_run=dry_run,
        repos=len(removed),
        refs=sum(len(refs) for refs in removed.values()),
    )
    return UnwatchResponse(removed=removed, dry_run=dry_run)


def handle_forget_stale(request: ForgetStaleRequest, index: Index) -> ForgetResponse:
    """Forget the repos whose checkout is gone."""
    gone = forget_stale_repos(index, dry_run=request.dry_run)
    return ForgetResponse(forgotten=[repo.repo_path for repo in gone], dry_run=request.dry_run)


def handle_forget(request: ForgetRequest, index: Index) -> ForgetResponse:
    """Forget one repo (metadata-only; GC reclaims the chunks).

    Only when its watch set is exactly `{HEAD}` — trim other refs
    first. `dry_run` reports without deleting.
    """
    target = index.find_repo(request.repo_path)
    if target is None:
        return ForgetResponse(forgotten=[], dry_run=request.dry_run)
    # Forget only when nothing beyond HEAD is watched (an empty watch set —
    # e.g. an inline `--no-daemon` index — also qualifies).
    if set(index.watched_refs(target)) - {HEAD_REF}:
        msg = "refusing to forget a repo watching refs beyond HEAD; remove them first"
        raise RbtrError(msg)
    if not request.dry_run:
        index.forget(target)
    return ForgetResponse(forgotten=[request.repo_path], dry_run=request.dry_run)


def handle_watch(
    request: WatchRequest,
    index: Index,
) -> WatchSetResponse:
    """Record the request's refs in the repo's watch set, and return the set.

    The worker derives and runs the actual build from `watched_refs`
    on its next poll.  `HEAD` is always included, so a repo first seen
    here watches it as one seen at startup does.  Dropping refs is
    `handle_unwatch`.
    """
    repo = index.register(request.repo_path)
    with index.session(repo) as ws:
        ws.add_watched_refs([HEAD_REF, *request.refs])
    log.info("watched_refs_added", repo=request.repo_path, refs=request.refs)
    return WatchSetResponse(watched=_watched_for_repo(index, repo))
