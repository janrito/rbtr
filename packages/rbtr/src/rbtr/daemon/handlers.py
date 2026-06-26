"""Request handlers for the daemon server.

Each handler takes a typed request and returns the success
response directly. Errors are signalled by raising `RbtrError`
(or subclasses like `IndexNotBuiltError`).

The daemon's `_dispatch` wraps every handler call in
`try/except Exception` and converts unhandled errors to
`ErrorResponse` for the protocol. CLI callers let exceptions
propagate to `main()`, which prints them.

Read handlers access the index via `store` (an `IndexStore`).
Write-side handlers (GC) also use `store`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import structlog

from rbtr.daemon.dto import RefOuts, SearchHitOut, SymbolOut
from rbtr.daemon.messages import (
    ActiveJob,
    BuildIndexRequest,
    ChangedSymbol,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    FindRefsRequest,
    FindRefsResponse,
    ForgetRequest,
    ForgetResponse,
    GcMode,
    GcRequest,
    GcResponse,
    IndexedRef,
    ListSymbolsRequest,
    ListSymbolsResponse,
    OkResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    Response,
    Scope,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
    WatchedRef,
)
from rbtr.errors import IndexNotBuiltError, RbtrError
from rbtr.git import (
    HEAD_REF,
    WORKTREE_REF,
    names_for_commits,
    normalise_repo_path,
    resolve_ref,
    worktree_tree_sha,
)
from rbtr.index.frames import changed_to_symbols
from rbtr.index.gc import run_gc, run_gc_all
from rbtr.index.models import Chunk, QueryKind, RepoRef

if TYPE_CHECKING:
    from rbtr.index.embeddings import Embedder
    from rbtr.index.reranker import Reranker
    from rbtr.index.store import IndexStore

log = structlog.get_logger(__name__)


def resolve_refs(repo_path: str, refs: list[str]) -> list[str]:
    """Resolve symbolic refs to commit SHAs (build path).

    Raises `RbtrError` if any ref cannot be resolved.
    """
    return [resolve_ref(repo_path, ref) for ref in refs]


def _resolve_read_ref(
    store: IndexStore,
    repo_path: str,
    repo_id: int,
    requested_ref: str | None,
    *,
    require_indexed: bool = False,
) -> str:
    """Resolve a ref for read operations.

    When *requested_ref* is `None`, checks the current worktree
    tree SHA via `worktree_tree_sha`.  If dirty and indexed,
    returns the tree SHA.  Otherwise resolves `"HEAD"`.  Explicit
    `"HEAD"` always resolves to the committed state.  Falls back
    to the latest indexed commit when the repo is missing.
    Raises `RbtrError` if the ref cannot be resolved.

    When *require_indexed* is set, the resolved SHA must be usable:

    - An explicit *requested_ref* whose SHA is not indexed is an
      error (`_require_indexed`) rather than a silent empty result.
    - The implicit worktree/HEAD path falls back to the latest
      indexed commit when the resolved SHA is not indexed (e.g. a
      build is still finalising), and only errors when the repo has
      no indexed commits at all. So an older indexed version of a
      symbol is preferred over an error.
    """
    explicit = requested_ref is not None
    if requested_ref is None:
        tree_sha = worktree_tree_sha(repo_path)
        if tree_sha is not None and store.has_indexed(repo_id, tree_sha):
            return tree_sha
        requested_ref = HEAD_REF
    try:
        sha = resolve_ref(repo_path, requested_ref)
    except RbtrError:
        if requested_ref == HEAD_REF:
            indexed = store.list_indexed_commits(repo_id)
            if indexed:
                return indexed[0][0]
        msg = f"Cannot resolve ref '{requested_ref}' in {repo_path}"
        raise RbtrError(msg) from None
    if require_indexed and not store.has_indexed(repo_id, sha):
        if not explicit:
            indexed = store.list_indexed_commits(repo_id)
            if indexed:
                return indexed[0][0]
        _require_indexed(store, repo_id, requested_ref, sha)
    return sha


# ── Read-only handlers ───────────────────────────────────────────────


def handle_search(
    request: SearchRequest,
    store: IndexStore,
    *,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> SearchResponse:
    """Search the index for `request.query`.

    `request.scope == Scope.ALL` searches every indexed repo and
    attributes each result with its `repo_path`; otherwise the
    search is scoped to the single repo at `request.repo_path`.

    Propagates `IndexNotBuiltError` from the store; the daemon's
    `_dispatch` turns it into an "index is building" message when a
    build is active.
    """
    if request.scope == Scope.ALL:
        refs = store.list_latest_refs()
        repo_paths = dict(store.list_repos())
    else:
        repo_id = store.resolve_repo(request.repo_path)
        ref = _resolve_read_ref(store, request.repo_path, repo_id, request.ref)
        refs = [RepoRef(repo_id=repo_id, commit_sha=ref)]
        repo_paths = None
    override = QueryKind(request.query_kind) if request.query_kind else None
    results = store.search(
        refs,
        request.query,
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
    ref = _resolve_read_ref(store, request.repo_path, repo_id, request.ref, require_indexed=True)
    chunks = store.match_by_name(ref, request.symbol, repo_id=repo_id)
    scoped = _scope_chunks(chunks, request.file_paths)
    return ReadSymbolResponse(chunks=[SymbolOut.from_chunk(c) for c in scoped])


def handle_list_symbols(request: ListSymbolsRequest, store: IndexStore) -> ListSymbolsResponse:
    repo_id = store.resolve_repo(request.repo_path)
    ref = _resolve_read_ref(store, request.repo_path, repo_id, request.ref, require_indexed=True)
    chunks = store.get_chunks(
        ref,
        file_path=request.file_path,
        repo_id=repo_id,
    )
    return ListSymbolsResponse(chunks=[SymbolOut.from_chunk(c) for c in chunks])


def handle_find_refs(request: FindRefsRequest, store: IndexStore) -> FindRefsResponse:
    repo_id = store.resolve_repo(request.repo_path)
    ref = _resolve_read_ref(store, request.repo_path, repo_id, request.ref, require_indexed=True)
    chunks = _scope_chunks(
        store.match_by_name(ref, request.symbol, repo_id=repo_id), request.file_paths
    )
    frame = store.inbound_refs(ref, [chunk.id for chunk in chunks], repo_id=repo_id)
    refs = RefOuts.validate_python(frame.to_dicts())
    return FindRefsResponse(refs=refs)


def _require_indexed(store: IndexStore, repo_id: int, requested_ref: str, sha: str) -> None:
    """Raise `IndexNotBuiltError` if *sha* is not indexed for this repo.

    The daemon's `_dispatch` upgrades this to an "index is building"
    message when a build is active; inline callers see the plain
    "not indexed" guidance.
    """
    if store.has_indexed(repo_id, sha):
        return
    if requested_ref == WORKTREE_REF:
        msg = "Working tree is not indexed yet — run rbtr index first"
    else:
        msg = f"Ref '{requested_ref}' is not indexed — run rbtr index first"
    raise IndexNotBuiltError(msg)


def handle_changed_symbols(
    request: ChangedSymbolsRequest, store: IndexStore
) -> ChangedSymbolsResponse:
    """Diff two indexed refs at the symbol level.

    Both refs must already be indexed; an unindexed side is an
    error rather than an empty diff (the symbol-level comparison
    reads chunks from both commits).
    """
    repo_id = store.resolve_repo(request.repo_path)
    base = resolve_ref(request.repo_path, request.base)
    head = resolve_ref(request.repo_path, request.head)
    _require_indexed(store, repo_id, request.base, base)
    _require_indexed(store, repo_id, request.head, head)
    frame = store.diff_symbols(base, head, repo_id=repo_id, file_paths=request.file_paths)
    changes = [
        ChangedSymbol(chunk=SymbolOut.from_chunk(chunk), change=change)
        for chunk, change in changed_to_symbols(frame)
    ]
    return ChangedSymbolsResponse(changes=changes)


type SnapshotStatusFn = Callable[[str], tuple[ActiveJob | None, ActiveJob | None]]


def _refs_for_repo(
    store: IndexStore,
    repo_id: int,
    repo_path: str,
) -> list[IndexedRef]:
    """Build `IndexedRef`s for one repo's indexed commits."""
    indexed_shas = [sha for sha, _ in store.list_indexed_commits(repo_id)]
    ref_names = names_for_commits(repo_path, indexed_shas)
    refs: list[IndexedRef] = []
    for sha in indexed_shas:
        total = store.count_chunks(sha, repo_id=repo_id)
        unembedded = store.count_unembedded(repo_id, sha)
        refs.append(
            IndexedRef(
                sha=sha,
                names=ref_names.get(sha, []),
                total=total,
                embedded=total - unembedded,
                repo_path=repo_path,
            )
        )
    return refs


def _watched_for_repo(
    store: IndexStore,
    repo_id: int,
    repo_path: str,
) -> list[WatchedRef]:
    """Build `WatchedRef`s for one repo's watch set.

    Resolves each watched ref to a SHA (`None` if it no longer
    resolves) and marks it indexed when that SHA is recorded in
    `indexed_commits`; otherwise it is pending.
    """
    out: list[WatchedRef] = []
    for ref in store.list_watched_refs(repo_id):
        try:
            sha: str | None = resolve_ref(repo_path, ref)
        except RbtrError:
            sha = None
        out.append(
            WatchedRef(
                ref=ref,
                sha=sha,
                indexed=sha is not None and store.has_indexed(repo_id, sha),
                repo_path=repo_path,
            )
        )
    return out


def handle_status(
    request: StatusRequest,
    store: IndexStore,
    snapshot_status: SnapshotStatusFn | None = None,
) -> StatusResponse:
    """Report index status for the workspace repo or every repo."""
    if request.scope == Scope.ALL:
        indexed_refs: list[IndexedRef] = []
        watched: list[WatchedRef] = []
        for repo_id, repo_path in store.list_repos():
            indexed_refs.extend(_refs_for_repo(store, repo_id, repo_path))
            watched.extend(_watched_for_repo(store, repo_id, repo_path))
    else:
        ws_repo_id = store.get_repo_id(request.repo_path)
        if ws_repo_id is None:
            return StatusResponse(
                db_path=store.db_path,
                indexed_refs=[],
                watched=[],
                active_build=None,
                active_embed=None,
            )
        indexed_refs = _refs_for_repo(store, ws_repo_id, request.repo_path)
        watched = _watched_for_repo(store, ws_repo_id, request.repo_path)
    active_build = None
    active_embed = None
    if snapshot_status is not None:
        active_build, active_embed = snapshot_status(request.repo_path)
    return StatusResponse(
        db_path=store.db_path,
        indexed_refs=indexed_refs,
        watched=watched,
        active_build=active_build,
        active_embed=active_embed,
    )


# ── Build handler ────────────────────────────────────────────────────


def handle_gc(request: GcRequest, store: IndexStore) -> GcResponse:
    t0 = time.monotonic()
    if request.repo_path is None:
        # Global GC: reclaim across every registered repo. Restricted to
        # the safe default reclamation — aggressive modes must be scoped to
        # one repo (a global drop of unwatched/non-HEAD commits is a
        # footgun, and KEEP refs are repo-specific).
        if request.mode is not GcMode.WATCHED:
            msg = (
                f"global GC supports only the default (watched) reclamation, "
                f"not {request.mode.value}; scope it with repo_path"
            )
            raise RbtrError(msg)
        counts, repos_collected = run_gc_all(
            store, mode=request.mode, refs=request.refs, dry_run=request.dry_run
        )
    else:
        repo_id = store.resolve_repo(request.repo_path)
        counts = run_gc(
            store,
            request.repo_path,
            repo_id,
            mode=request.mode,
            refs=request.refs,
            dry_run=request.dry_run,
        )
        repos_collected = 1
    # A real run's `counts.chunks` is the actual reclamation. A dry run only
    # predicts the drop set's freed chunks, so add the pre-existing orphans
    # the global prune would also remove (counted once across all repos).
    chunks_freed = counts.chunks
    if request.dry_run:
        chunks_freed += store.count_orphan_chunks()
    elapsed = time.monotonic() - t0
    log.info(
        "gc_complete",
        mode=request.mode,
        dry_run=request.dry_run,
        repos=repos_collected,
        commits=counts.commits,
        snapshots=counts.snapshots,
        edges=counts.edges,
        chunks_freed=chunks_freed,
        elapsed_ms=round(elapsed * 1000, 1),
    )
    return GcResponse(
        repos_collected=repos_collected,
        commits_dropped=counts.commits,
        snapshots_dropped=counts.snapshots,
        edges_dropped=counts.edges,
        chunks_freed=chunks_freed,
        elapsed_seconds=elapsed,
        dry_run=request.dry_run,
    )


def handle_forget(request: ForgetRequest, store: IndexStore) -> ForgetResponse:
    """Forget whole repos (metadata-only; GC reclaims the chunks).

    `stale=True`: forget every registered repo whose stored path no longer
    resolves (a removed worktree/clone) — found by enumeration, since a
    gone path cannot be normalised into a request. Otherwise forget the
    single `repo_path`, but only when its watch set is exactly `{HEAD}`
    (trim other refs first). `dry_run` reports without deleting.
    """
    if request.stale:
        gone: list[tuple[int, str]] = []
        for repo_id, path in store.list_repos():
            try:
                normalise_repo_path(path)
            except RbtrError:
                gone.append((repo_id, path))
        if not request.dry_run and gone:
            with store.session() as ws:
                for repo_id, _path in gone:
                    ws.forget_repo(repo_id)
        return ForgetResponse(forgotten=[path for _id, path in gone], dry_run=request.dry_run)

    if request.repo_path is None:
        msg = "forget requires a repo_path or stale=True"
        raise RbtrError(msg)
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


class WatchFn(Protocol):
    """The watch-set mutation the index handler is given (injected).

    Implemented by `DaemonServer.watch_refs`; a `Protocol` (not
    `Callable[..., None]`) so the keyword-only `remove` stays typed.
    """

    def __call__(self, repo_path: str, refs: list[str], *, remove: bool) -> None: ...


def handle_build_index(
    request: BuildIndexRequest,
    watch: WatchFn,
) -> Response:
    """Record (or remove) the request's refs in the repo's watch set.

    The worker derives and runs the actual build from `watched_refs`
    on its next poll. `remove=True` stops watching the given refs;
    `HEAD` cannot be removed (the daemon rejects it atomically).
    """
    watch(request.repo_path, request.refs, remove=request.remove)
    return OkResponse()
