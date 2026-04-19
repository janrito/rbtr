"""Request handlers for the daemon server.

Each handler takes a typed request and returns a typed response.
All index operations are routed through the `RepoManager` to
resolve repo paths to `repo_id`s.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pygit2

from rbtr.daemon.messages import (
    BuildIndexRequest,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    ErrorCode,
    ErrorResponse,
    FindRefsRequest,
    FindRefsResponse,
    GcRequest,
    GcResponse,
    ListSymbolsRequest,
    ListSymbolsResponse,
    OkResponse,
    QueueItem,
    ReadSymbolRequest,
    ReadSymbolResponse,
    Response,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.repos import RepoManager
from rbtr.errors import RbtrError
from rbtr.git import changed_files, names_for_commits, open_repo, resolve_commit
from rbtr.index.gc import run_gc

if TYPE_CHECKING:
    from rbtr.daemon.build_queue import BuildQueue

log = logging.getLogger(__name__)


def resolve_refs(repo: pygit2.Repository, refs: list[str]) -> list[str] | ErrorResponse:
    """Resolve symbolic refs to commit SHAs. Returns error on failure."""
    resolved: list[str] = []
    for ref in refs:
        try:
            sha = str(resolve_commit(repo, ref).id)
        except KeyError as exc:
            return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))
        resolved.append(sha)
    return resolved


_HEX_SHA_LEN = 40


def _looks_like_sha(ref: str) -> bool:
    """Return True if *ref* is a full 40-char hex SHA.

    Used by the read-side handlers to skip git resolution when the
    caller already supplied a resolved commit ID.
    """
    return len(ref) == _HEX_SHA_LEN and all(c in "0123456789abcdef" for c in ref.lower())


def _resolve_read_ref(
    mgr: RepoManager,
    repo_path: str,
    repo_id: int,
    requested_ref: str,
) -> str | ErrorResponse:
    """Resolve *requested_ref* for a read-side handler.

    The read path doesn't need to walk the working tree; it only
    needs a commit SHA to scope the query.  Try, in order:

    1. If *requested_ref* is already a full SHA, use it as-is.
    2. Open the repo with `pygit2` and resolve the symbolic ref.
    3. If `open_repo` fails AND *requested_ref* is "HEAD"
       AND the store knows at least one indexed commit, use the
       most recently indexed one.  This makes the daemon serve
       the latest index even when the caller's checkout has
       moved or is missing -- desirable for read-only queries.

    Step 3 is *only* taken when the repo cannot be opened; an
    open repo with no resolvable HEAD still propagates the error
    so callers see real git problems.
    """
    if _looks_like_sha(requested_ref):
        return requested_ref
    try:
        repo = open_repo(repo_path)
    except RbtrError as exc:
        if requested_ref == "HEAD":
            indexed = mgr.store.list_indexed_commits(repo_id)
            if indexed:
                return indexed[0][0]
        return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))
    try:
        return str(resolve_commit(repo, requested_ref).id)
    except KeyError as exc:
        return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))


# ── Read-only handlers ───────────────────────────────────────────────


def handle_search(request: SearchRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    ref = _resolve_read_ref(mgr, request.repo, repo_id, request.ref)
    if isinstance(ref, ErrorResponse):
        return ref
    results = mgr.store.search(
        ref,
        request.query,
        top_k=request.limit,
        variant=request.variant,
        alpha=request.alpha,
        beta=request.beta,
        gamma=request.gamma,
        repo_id=repo_id,
    )
    return SearchResponse(results=results)


def handle_read_symbol(request: ReadSymbolRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    ref = _resolve_read_ref(mgr, request.repo, repo_id, request.ref)
    if isinstance(ref, ErrorResponse):
        return ref
    chunks = mgr.store.search_by_name(ref, request.name, variant=request.variant, repo_id=repo_id)
    return ReadSymbolResponse(chunks=chunks)


def handle_list_symbols(request: ListSymbolsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    ref = _resolve_read_ref(mgr, request.repo, repo_id, request.ref)
    if isinstance(ref, ErrorResponse):
        return ref
    chunks = mgr.store.get_chunks(
        ref,
        file_path=request.file_path,
        variant=request.variant,
        repo_id=repo_id,
    )
    return ListSymbolsResponse(chunks=chunks)


def handle_find_refs(request: FindRefsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    ref = _resolve_read_ref(mgr, request.repo, repo_id, request.ref)
    if isinstance(ref, ErrorResponse):
        return ref
    edges = mgr.store.get_edges(ref, target_id=request.symbol, repo_id=repo_id)
    return FindRefsResponse(edges=edges)


def handle_changed_symbols(request: ChangedSymbolsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    repo = open_repo(request.repo)
    base = str(resolve_commit(repo, request.base).id)
    head = str(resolve_commit(repo, request.head).id)
    changed = changed_files(repo, base, head)
    chunks = []
    for path in sorted(changed):
        chunks.extend(
            mgr.store.get_chunks(head, file_path=path, variant=request.variant, repo_id=repo_id)
        )
    return ChangedSymbolsResponse(chunks=chunks)


def handle_status(
    request: StatusRequest,
    mgr: RepoManager,
    build_queue: BuildQueue | None,
) -> Response:
    repo_id = mgr.resolve(request.repo)
    indexed_refs = [sha for sha, _ in mgr.store.list_indexed_commits(repo_id)]
    # Resolve HEAD via git when possible; degrade gracefully to the
    # latest indexed commit when the repo is missing (same policy as
    # the read-side handlers above — status is a read).
    head: str | None
    indexed_ref_names: dict[str, list[str]] = {}
    try:
        repo = open_repo(request.repo)
    except RbtrError:
        head = indexed_refs[0] if indexed_refs else None
    else:
        try:
            head = str(resolve_commit(repo, "HEAD").id)
        except KeyError:
            head = indexed_refs[0] if indexed_refs else None
        indexed_ref_names = names_for_commits(repo, indexed_refs)
    count = mgr.store.count_chunks(head, repo_id=repo_id) if head is not None else 0
    active_job = None
    pending: list[QueueItem] = []
    if build_queue is not None:
        active_job, pending = build_queue.snapshot_status()
    return StatusResponse(
        exists=count > 0,
        db_path=mgr.store.db_path,
        total_chunks=count,
        indexed_refs=indexed_refs,
        indexed_ref_names=indexed_ref_names,
        active_job=active_job,
        pending=pending,
    )


# ── Build handler ────────────────────────────────────────────────────


def handle_gc(request: GcRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    repo = open_repo(request.repo)
    t0 = time.monotonic()
    try:
        counts = run_gc(
            mgr.store,
            repo,
            repo_id,
            mode=request.mode,
            refs=request.refs,
            dry_run=request.dry_run,
        )
    except RbtrError as exc:
        return ErrorResponse(code=ErrorCode.INVALID_REQUEST, message=str(exc))
    return GcResponse(
        commits_dropped=counts.commits,
        snapshots_dropped=counts.snapshots,
        edges_dropped=counts.edges,
        chunks_dropped=counts.chunks,
        elapsed_seconds=time.monotonic() - t0,
        dry_run=request.dry_run,
    )


def handle_build_index(
    request: BuildIndexRequest,
    build_queue: object,
) -> Response:
    """Submit a build to the queue.

    The repo row is created (or fetched) by the build queue's own
    call to ``RepoManager.resolve``, so no separate registration is
    needed here. The watcher learns about the repo on its next poll
    via ``store.list_repos``.
    """
    from rbtr.daemon.build_queue import BuildQueue

    if not isinstance(build_queue, BuildQueue):
        return ErrorResponse(code=ErrorCode.INTERNAL, message="No build queue")

    build_queue.submit(
        request.repo,
        request.refs,
        strip_docstrings=request.strip_docstrings,
    )
    return OkResponse()
