"""Request handlers for the daemon server.

Each handler takes a typed request and returns a typed response.
All index operations are routed through the `RepoManager` to
resolve repo paths to `repo_id`s.
"""

from __future__ import annotations

import logging
import time

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
from rbtr.git import changed_files, open_repo, resolve_commit
from rbtr.index.gc import run_gc

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


# ── Read-only handlers ───────────────────────────────────────────────


def handle_search(request: SearchRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    results = mgr.store.search(
        request.ref,
        request.query,
        top_k=request.limit,
        repo_id=repo_id,
    )
    return SearchResponse(results=results)


def handle_read_symbol(request: ReadSymbolRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    chunks = mgr.store.search_by_name(request.ref, request.name, repo_id=repo_id)
    return ReadSymbolResponse(chunks=chunks)


def handle_list_symbols(request: ListSymbolsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    chunks = mgr.store.get_chunks(request.ref, file_path=request.file_path, repo_id=repo_id)
    return ListSymbolsResponse(chunks=chunks)


def handle_find_refs(request: FindRefsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    edges = mgr.store.get_edges(request.ref, target_id=request.symbol, repo_id=repo_id)
    return FindRefsResponse(edges=edges)


def handle_changed_symbols(request: ChangedSymbolsRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    repo = open_repo(request.repo)
    changed = changed_files(repo, request.base, request.head)
    chunks = []
    for path in sorted(changed):
        chunks.extend(mgr.store.get_chunks(request.head, file_path=path, repo_id=repo_id))
    return ChangedSymbolsResponse(chunks=chunks)


def handle_status(request: StatusRequest, mgr: RepoManager) -> Response:
    repo_id = mgr.resolve(request.repo)
    count = mgr.store.count_chunks("HEAD", repo_id=repo_id)
    indexed_refs = [sha for sha, _ in mgr.store.list_indexed_commits(repo_id)]
    return StatusResponse(
        exists=count > 0,
        db_path=mgr.store.db_path,
        total_chunks=count,
        indexed_refs=indexed_refs,
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

    build_queue.submit(request.repo, request.refs)
    return OkResponse()
