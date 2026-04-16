"""Request handlers for the daemon server.

Each handler takes a typed request and returns a typed response.
All index operations are routed through the `RepoManager` to
resolve repo paths to `repo_id`s.
"""

from __future__ import annotations

import logging

import pygit2

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
from rbtr.git import changed_files, open_repo, resolve_commit
from rbtr.index.orchestrator import build_index

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
    count = mgr.store.count_chunks("HEAD", repo_id=repo_id)
    return StatusResponse(
        exists=count > 0,
        db_path=mgr.store.db_path,
        total_chunks=count,
    )


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


def handle_build_index(request: BuildIndexRequest, mgr: RepoManager) -> Response:
    """Index refs for a repo. Runs synchronously.

    Each ref is resolved to a commit SHA before indexing.
    Blob dedup means indexing multiple refs only extracts
    files that differ between them.
    """
    repo_id = mgr.resolve(request.repo)
    try:
        repo = open_repo(request.repo)
    except Exception as exc:
        return ErrorResponse(code=ErrorCode.REPO_NOT_FOUND, message=str(exc))

    resolved = resolve_refs(repo, request.refs)
    if isinstance(resolved, ErrorResponse):
        return resolved

    result = None
    for sha in resolved:
        result = build_index(repo, sha, mgr.store, repo_id=repo_id)

    if result is None:
        return ErrorResponse(
            code=ErrorCode.INVALID_REQUEST,
            message="No refs to index",
        )

    return BuildIndexResponse(
        refs=resolved,
        stats=result.stats,
        errors=result.errors,
    )
