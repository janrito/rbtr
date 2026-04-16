"""Request handlers for the daemon server.

Each handler takes a typed request and returns a typed response.
All index operations are routed through the `RepoManager` to
resolve repo paths to `repo_id`s.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import pygit2

from rbtr.daemon.messages import (
    BuildIndexRequest,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    ErrorCode,
    ErrorResponse,
    FindRefsRequest,
    FindRefsResponse,
    IndexErrorNotification,
    ListSymbolsRequest,
    ListSymbolsResponse,
    Notification,
    OkResponse,
    ProgressNotification,
    ReadSymbolRequest,
    ReadSymbolResponse,
    ReadyNotification,
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

# Track which repos are currently building (thread-safe via GIL)
_building: set[str] = set()


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


# ── Build handler (async — returns immediately) ─────────────────────

type NotifyFn = Callable[[Notification], None]


def handle_build_index_async(
    request: BuildIndexRequest,
    mgr: RepoManager,
    notify: NotifyFn,
) -> Response:
    """Start an async index build. Returns immediately.

    Progress and completion are published via *notify*.
    """
    if request.repo in _building:
        return ErrorResponse(
            code=ErrorCode.INDEX_IN_PROGRESS,
            message=f"Build already in progress for {request.repo}",
        )

    thread = threading.Thread(
        target=_do_build,
        args=(request.repo, request.refs, mgr, notify),
        daemon=True,
    )
    thread.start()
    return OkResponse()


def _do_build(
    repo_path: str,
    refs: list[str],
    mgr: RepoManager,
    notify: NotifyFn,
) -> None:
    """Run build_index in a thread. Publishes notifications."""
    _building.add(repo_path)
    try:
        repo = open_repo(repo_path)
        repo_id = mgr.resolve(repo_path)

        resolved = resolve_refs(repo, refs)
        if isinstance(resolved, ErrorResponse):
            notify(IndexErrorNotification(repo=repo_path, message=resolved.message))
            return

        for sha in resolved:

            def on_progress(done: int, total: int) -> None:
                notify(
                    ProgressNotification(
                        repo=repo_path,
                        phase="parsing",
                        current=done,
                        total=total,
                    )
                )

            def on_embed_progress(done: int, total: int) -> None:
                notify(
                    ProgressNotification(
                        repo=repo_path,
                        phase="embedding",
                        current=done,
                        total=total,
                    )
                )

            result = build_index(
                repo,
                sha,
                mgr.store,
                repo_id=repo_id,
                on_progress=on_progress,
                on_embed_progress=on_embed_progress,
            )

            notify(
                ReadyNotification(
                    repo=repo_path,
                    ref=sha,
                    chunks=result.stats.total_chunks,
                    edges=result.stats.total_edges,
                    elapsed=round(result.stats.elapsed_seconds, 2),
                )
            )

    except Exception as exc:
        log.exception("Build failed for %s", repo_path)
        notify(IndexErrorNotification(repo=repo_path, message=str(exc)))
    finally:
        _building.discard(repo_path)
