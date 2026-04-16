"""Request handlers for the daemon server.

Each handler takes a typed request and returns a typed response.
All index operations are routed through the `RepoManager` to
resolve repo paths to `repo_id`s.
"""

from __future__ import annotations

import logging

from rbtr.daemon.messages import (
    BuildIndexRequest,
    BuildIndexResponse,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    ErrorCode,
    ErrorResponse,
    FindRefsRequest,
    FindRefsResponse,
    ListSymbolsRequest,
    ListSymbolsResponse,
    ReadSymbolRequest,
    ReadSymbolResponse,
    Response,
    SearchRequest,
    SearchResponse,
    StatusRequest,
    StatusResponse,
)
from rbtr.daemon.repos import RepoManager
from rbtr.git import changed_files, open_repo
from rbtr.index.orchestrator import build_index, update_index

log = logging.getLogger(__name__)


def handle_search(request: SearchRequest, mgr: RepoManager) -> Response:
    """Search the index for a repo."""
    repo_id = mgr.resolve(request.repo)
    results = mgr.store.search(
        request.ref,
        request.query,
        top_k=request.limit,
        repo_id=repo_id,
    )
    return SearchResponse(results=results)


def handle_read_symbol(request: ReadSymbolRequest, mgr: RepoManager) -> Response:
    """Read symbols matching a name pattern."""
    repo_id = mgr.resolve(request.repo)
    chunks = mgr.store.search_by_name(request.ref, request.name, repo_id=repo_id)
    return ReadSymbolResponse(chunks=chunks)


def handle_list_symbols(request: ListSymbolsRequest, mgr: RepoManager) -> Response:
    """List symbols in a file."""
    repo_id = mgr.resolve(request.repo)
    chunks = mgr.store.get_chunks(request.ref, file_path=request.file_path, repo_id=repo_id)
    return ListSymbolsResponse(chunks=chunks)


def handle_find_refs(request: FindRefsRequest, mgr: RepoManager) -> Response:
    """Find edges referencing a symbol."""
    repo_id = mgr.resolve(request.repo)
    edges = mgr.store.get_edges(request.ref, target_id=request.symbol, repo_id=repo_id)
    return FindRefsResponse(edges=edges)


def handle_changed_symbols(request: ChangedSymbolsRequest, mgr: RepoManager) -> Response:
    """List symbols in files that changed between two refs."""
    repo_id = mgr.resolve(request.repo)
    repo = open_repo(request.repo)
    changed = changed_files(repo, request.base, request.head)
    chunks = []
    for path in sorted(changed):
        chunks.extend(mgr.store.get_chunks(request.head, file_path=path, repo_id=repo_id))
    return ChangedSymbolsResponse(chunks=chunks)


def handle_status(request: StatusRequest, mgr: RepoManager) -> Response:
    """Check index status for a repo."""
    repo_id = mgr.resolve(request.repo)
    chunks = mgr.store.get_chunks("HEAD", repo_id=repo_id)
    return StatusResponse(
        exists=len(chunks) > 0,
        db_path=mgr.store.db_path,
        total_chunks=len(chunks),
    )


def handle_build_index(request: BuildIndexRequest, mgr: RepoManager) -> Response:
    """Index one or two refs for a repo.

    Single ref: full index at that commit.
    Two refs: index base, then incremental update for head.
    """
    repo_id = mgr.resolve(request.repo)
    try:
        repo = open_repo(request.repo)
    except Exception as exc:
        return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))

    refs = request.refs
    if len(refs) == 1:
        result = build_index(repo, refs[0], mgr.store, repo_id=repo_id)
    elif len(refs) == 2:
        build_index(repo, refs[0], mgr.store, repo_id=repo_id)
        result = update_index(repo, refs[0], refs[1], mgr.store, repo_id=repo_id)
    else:
        return ErrorResponse(
            code=ErrorCode.INVALID_REQUEST,
            message=f"Expected 1 or 2 refs, got {len(refs)}",
        )

    return BuildIndexResponse(
        refs=refs,
        stats=result.stats,
        errors=result.errors,
    )
