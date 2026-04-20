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
from rbtr.git import (
    changed_files,
    names_for_commits,
    open_repo,
    resolve_build_ref,
    resolve_read_ref,
)
from rbtr.index.gc import run_gc

if TYPE_CHECKING:
    from rbtr.daemon.build_queue import BuildQueue

log = logging.getLogger(__name__)


def resolve_refs(repo: pygit2.Repository, refs: list[str]) -> list[str] | ErrorResponse:
    """Resolve symbolic refs to commit SHAs (build path).

    Returns either the resolved list or an `ErrorResponse` for the
    daemon-protocol caller.  Wraps `resolve_build_ref` and converts
    its `RbtrError` into the typed protocol error.
    """
    resolved: list[str] = []
    for ref in refs:
        try:
            resolved.append(resolve_build_ref(repo, ref))
        except RbtrError as exc:
            return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))
    return resolved


def _read_ref_or_error(
    mgr: RepoManager,
    repo_path: str,
    repo_id: int,
    requested_ref: str,
) -> str | ErrorResponse:
    """Daemon-side wrapper around `resolve_read_ref`.

    Returns the resolved SHA, or an `ErrorResponse` with code
    `REPO_NOT_FOUND` so the caller can return it directly.  Bridges
    the typed-error daemon protocol to the protocol-agnostic git
    helper.
    """

    def _latest() -> str | None:
        indexed = mgr.store.list_indexed_commits(repo_id)
        return indexed[0][0] if indexed else None

    sha = resolve_read_ref(repo_path, requested_ref, latest_indexed=_latest)
    if sha is None:
        msg = f"Cannot resolve ref '{requested_ref}' in {repo_path}"
        return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=msg)
    return sha


# ── Read-only handlers ───────────────────────────────────────────────


def handle_search(request: SearchRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    ref = _read_ref_or_error(mgr, request.repo, repo_id, request.ref)
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
    ref = _read_ref_or_error(mgr, request.repo, repo_id, request.ref)
    if isinstance(ref, ErrorResponse):
        return ref
    chunks = mgr.store.search_by_name(ref, request.name, variant=request.variant, repo_id=repo_id)
    return ReadSymbolResponse(chunks=chunks)


def handle_list_symbols(request: ListSymbolsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    ref = _read_ref_or_error(mgr, request.repo, repo_id, request.ref)
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
    ref = _read_ref_or_error(mgr, request.repo, repo_id, request.ref)
    if isinstance(ref, ErrorResponse):
        return ref
    edges = mgr.store.get_edges(ref, target_id=request.symbol, repo_id=repo_id)
    return FindRefsResponse(edges=edges)


def handle_changed_symbols(request: ChangedSymbolsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    repo = open_repo(request.repo)
    base = resolve_build_ref(repo, request.base)
    head = resolve_build_ref(repo, request.head)
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
    # `head` uses the standard read-ref policy: SHA short-circuit,
    # then pygit2, then fall back to the latest indexed commit when
    # the working tree is missing.  `indexed_ref_names` is a separate
    # concern -- it needs an open repo, so it stays inline.
    head = resolve_read_ref(
        request.repo,
        "HEAD",
        latest_indexed=lambda: indexed_refs[0] if indexed_refs else None,
    )
    indexed_ref_names: dict[str, list[str]] = {}
    try:
        repo = open_repo(request.repo)
    except RbtrError:
        pass
    else:
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
