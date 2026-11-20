"""Index building — extract chunks and infer edges for a commit.

`build_index` is the entry point.  It runs three phases, each in its own
transactional session:

1. **Extract** — stream files from git, extract chunks via tree-sitter or
   plugin chunkers, write chunks + snapshots.
2. **Edges** — infer import/test/doc edges from the committed chunk set.
3. **Finalise** — mark the commit indexed, clean up orphans.

Embedding is a separate step — see `embed_index` in `rbtr.index.embed`.

All heavy work runs synchronously — the caller (daemon job worker) runs it
via `asyncio.to_thread()`.  Progress is reported via a single
`ProgressCallback(phase, done, total)`; logs record completion summaries.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import structlog

from rbtr.config import config
from rbtr.domain.edges import infer_edges
from rbtr.domain.models import (
    FileOutcome,
    FileSnapshot,
    IndexResult,
    Repo,
)
from rbtr.errors import BuildStopped
from rbtr.git import FileEntry, changed_files, list_files, non_commit_shas, normalise_repo_path
from rbtr.index.index import Index
from rbtr.index.progress import ProgressCallback, _noop_progress
from rbtr.index.staging import serial_map_view, staged_chunks
from rbtr.languages.extract import BATCH_FILES, extract_files, file_rows
from rbtr.languages.manager import get_manager
from rbtr.languages.plaintext import PLAINTEXT
from rbtr.languages.resolution import language_resolutions
from rbtr.rbtrignore import load_ignore

log = structlog.get_logger(__name__)


def drop_stale_worktree_shas(index: Index, repo: Repo, *, keep: str) -> None:
    """Drop other worktree snapshots of repo, retaining keep in finite sessions."""
    indexed = [s.snapshot_sha for s in index.indexed_snapshots(repo)]
    stale = [sha for sha in non_commit_shas(repo.repo_path, indexed) if sha != keep]
    if not stale:
        return
    with index.session(repo) as ws:
        for sha in stale:
            ws.drop_snapshot(snapshot_sha=sha)
            log.info("dropped_stale_worktree_sha", sha=sha[:12])


@dataclass(frozen=True)
class StoredFile:
    """A listed file whose chunks are already stored: unchanged since the base snapshot, or its blob current.

    *outcome* is which: `SKIPPED_UNCHANGED` or `SKIPPED_CURRENT`. The
    file still gets its snapshot row, under *language*.
    """

    entry: FileEntry
    language: str
    outcome: FileOutcome


@dataclass(frozen=True)
class FileToExtract:
    """A listed file the build extracts as *language*.

    *language* is both its chunks' `file_language`, part of their
    identity, and its snapshot row's `detected_language`, which is how a
    read pairs the two: holding one value stops them disagreeing, which
    would leave the file indexed and unreachable.
    """

    entry: FileEntry
    language: str


def file_languages(
    files: Sequence[FileEntry], stored: dict[str, str], changed: set[str] | None
) -> list[tuple[FileEntry, str]]:
    """Each file with its language: extension, stored detection, then content.

    Files outside *changed* keep their extension or stored language;
    content detection applies to new or changed files. Plaintext is the
    fallback when no language claims a file.
    """
    mgr = get_manager()
    resolved: list[tuple[FileEntry, str]] = []
    for entry in files:
        language = mgr.detect_language(entry.path) or stored.get(entry.path, "")
        if not language and (changed is None or entry.path in changed):
            text = entry.content.decode(errors="replace")
            language = mgr.detect_language_by_content(text) or ""
        resolved.append((entry, language or PLAINTEXT))
    return resolved


def listed_files(
    files: Sequence[tuple[FileEntry, str]],
    current: set[tuple[str, str]],
    changed: set[str] | None,
) -> list[StoredFile | FileToExtract]:
    """Classify resolved files using their current extractions and changed paths.

    A file outside *changed* is unchanged. Otherwise its blob must have
    a current extraction in the resolved language to be stored already.
    """
    listed: list[StoredFile | FileToExtract] = []
    for entry, language in files:
        if changed is not None and entry.path not in changed:
            listed.append(StoredFile(entry, language, FileOutcome.SKIPPED_UNCHANGED))
        elif (entry.blob_sha, language) in current:
            listed.append(StoredFile(entry, language, FileOutcome.SKIPPED_CURRENT))
        else:
            listed.append(FileToExtract(entry, language))
    return listed


@dataclass(frozen=True, slots=True)
class _Extracted:
    """What extraction leaves behind for the edge pass.

    The edge pass reads no files, so anything it needs from their content
    has to be carried here: *manifests* holds the text of the files
    languages declare as their `manifest`.
    """

    result: IndexResult
    repo_files: set[str]
    manifests: dict[str, str]


def _extract_and_store_chunks(
    *,
    index: Index,
    repo_path: str,
    snapshot_sha: str,
    repo: Repo,
    base_sha: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
    stop: threading.Event | None = None,
) -> _Extracted:
    """Stream files, extract chunks, write snapshots.

    When *base_sha* is provided, only files that changed between
    *base_sha* and *snapshot_sha* are extracted.
    """
    mgr = get_manager()
    # Each chunk is stamped with its OWN language's extraction serial, not
    # the host file's. They coincide for single-language files; they differ
    # for multi-language files (SFCs), whose delegated chunks carry an
    # embedded language. Built once from the registry.
    serial_by_language = {
        lang: reg.extraction_serial
        for lang in mgr.all_language_ids()
        if (reg := mgr.get_registration(lang)) is not None
    }
    serials = serial_map_view(serial_by_language)
    repo_root = Path(repo_path).resolve()
    ignore = load_ignore(repo_root)
    changed: set[str] | None = None
    if base_sha is not None:
        changed = changed_files(repo_path, base_sha, snapshot_sha)

    snapshots: list[FileSnapshot] = []
    result = IndexResult()
    repo_files: set[str] = set()  # collected for edge inference
    # Files a language declares as its manifest, read while their content is
    # in hand: resolution needs them, and the edge pass sees only paths.
    manifest_names = {
        reg.manifest
        for lang_id in mgr.all_language_ids()
        if (reg := mgr.get_registration(lang_id)) is not None and reg.manifest
    }
    manifests: dict[str, str] = {}

    with index.session(repo) as session:
        # Stream files from git.  A file's content is held until its
        # batch is stored, then released.
        files = list_files(
            repo_path,
            snapshot_sha,
            max_file_size=config.max_file_size,
            ignore=ignore,
        )
        for entries in itertools.batched(files, BATCH_FILES, strict=False):
            if stop is not None and stop.is_set():
                raise BuildStopped
            stored = session.snapshot_languages([entry.path for entry in entries])
            resolved = file_languages(entries, stored, changed)
            current = session.current_blobs(
                [
                    entry.blob_sha
                    for entry, _ in resolved
                    if changed is None or entry.path in changed
                ],
                serials,
            )
            batch = listed_files(resolved, current, changed)
            to_extract = [(f.entry, f.language) for f in batch if isinstance(f, FileToExtract)]
            failed: list[str] = []
            if to_extract:
                chunks, failed = extract_files(file_rows(to_extract))
                session.replace_chunks(staged_chunks(chunks, serial_by_language))

            for listed_file in batch:
                if stop is not None and stop.is_set():
                    raise BuildStopped
                entry = listed_file.entry
                repo_files.add(entry.path)
                if entry.path in manifest_names:
                    manifests[entry.path] = entry.content.decode(errors="replace")
                match listed_file:
                    case StoredFile(outcome=outcome):
                        result.stats.record(outcome)
                    case FileToExtract() if entry.path in failed:
                        result.errors.append(f"Failed to index {entry.path}")
                        result.stats.record(FileOutcome.FAILED)
                    case FileToExtract():
                        result.stats.record(FileOutcome.PARSED)
                snapshots.append(
                    FileSnapshot(
                        snapshot_sha=snapshot_sha,
                        file_path=entry.path,
                        blob_sha=entry.blob_sha,
                        detected_language=listed_file.language,
                    )
                )
                on_progress("parsing", result.stats.total_files, result.stats.total_files)

        session.replace_snapshots(snapshots, snapshot_sha=snapshot_sha)

    # Every outcome that occurred, by name, so a run's file count is
    # accounted for in the log line that reports it.
    log.info(
        "extracted_files",
        total=result.stats.total_files,
        sha=snapshot_sha[:12],
        **{outcome.value: count for outcome, count in result.stats.outcomes.items()},
    )
    return _Extracted(result=result, repo_files=repo_files, manifests=manifests)


def _infer_and_store_edges(
    *,
    index: Index,
    repo_files: set[str],
    manifests: dict[str, str],
    snapshot_sha: str,
    repo: Repo,
    on_progress: ProgressCallback,
) -> tuple[int, int]:
    """Infer the snapshot's edges and replace its stored ones; return edge and chunk counts."""
    on_progress("edges", 0, 0)
    resolutions = language_resolutions(get_manager().registrations(), manifests=manifests)
    with index.session(repo) as session:
        chunks = session.edge_chunks(snapshot_sha)
        edges = infer_edges(chunks, repo_files, resolutions)
        session.replace_edges(edges, snapshot_sha=snapshot_sha)
    log.info("inferred_edges", edges=len(edges))
    return len(edges), len(chunks)


def _mark_indexed_and_cleanup(
    *, index: Index, repo: Repo, snapshot_sha: str, on_progress: ProgressCallback
) -> None:
    """Mark the commit indexed and remove orphaned data."""
    on_progress("finalising", 0, 0)
    with index.session(repo) as session:
        session.mark_indexed(snapshot_sha=snapshot_sha)
        cleaned = session.cleanup()
        if cleaned.file_snapshots or cleaned.edges or cleaned.chunks:
            log.info(
                "cleanup",
                file_snapshots=cleaned.file_snapshots,
                edges=cleaned.edges,
                chunks=cleaned.chunks,
            )


def build_index(
    repo_path: str,
    snapshot_sha: str,
    index: Index,
    *,
    base_sha: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
    stop: threading.Event | None = None,
) -> IndexResult:
    """Build (or incrementally update) the index for *snapshot_sha*.

    Lists all files at *snapshot_sha*, extracts chunks, infers
    edges, and marks the commit indexed.  The commit becomes
    queryable via FTS/name/edges immediately — embedding is
    handled separately by `embed_index`.

    *repo_path* may be any path inside the repository; it is resolved
    to the root and registered if not already known, since indexing a
    path is what puts a repo in the index.  The repo comes from
    that lookup rather than from the caller, so it always matches the
    path being built.  A linked worktree resolves to its own
    directory, and so is built as a repo of its own, in its main
    checkout's project.

    When *base_sha* is provided, only files that changed between
    *base_sha* and *snapshot_sha* are considered for extraction.
    Unchanged files are skipped whatever the blob check says.

    When *stop* is set, the build raises `BuildStopped` before its next
    file or phase. The snapshot is left unindexed, as a build that died
    there would leave it, and the next build of it starts again.
    """
    t0 = time.monotonic()

    # Register before writing anything that refers to the repo, so a
    # crash part-way through leaves an unused repos row rather than
    # indexed rows that nothing can find.
    repo_path = normalise_repo_path(repo_path)
    repo = index.register(repo_path)

    # Phase 1: extract chunks from git, write to DB.
    extracted = _extract_and_store_chunks(
        index=index,
        repo_path=repo_path,
        snapshot_sha=snapshot_sha,
        repo=repo,
        base_sha=base_sha,
        on_progress=on_progress,
        stop=stop,
    )
    if stop is not None and stop.is_set():
        raise BuildStopped

    result = extracted.result

    # Phase 2: infer cross-file edges.
    result.stats.total_edges, result.stats.total_chunks = _infer_and_store_edges(
        index=index,
        repo_files=extracted.repo_files,
        manifests=extracted.manifests,
        snapshot_sha=snapshot_sha,
        repo=repo,
        on_progress=on_progress,
    )
    if stop is not None and stop.is_set():
        raise BuildStopped

    # Phase 3: mark complete and remove orphaned data.
    _mark_indexed_and_cleanup(
        index=index, repo=repo, snapshot_sha=snapshot_sha, on_progress=on_progress
    )

    result.stats.elapsed_seconds = time.monotonic() - t0
    log.info(
        "index_complete",
        chunks=result.stats.total_chunks,
        edges=result.stats.total_edges,
        elapsed_seconds=round(result.stats.elapsed_seconds, 1),
    )
    return result
