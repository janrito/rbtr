"""Index building — extract chunks and infer edges for a commit.

`build_index` is the entry point.  It indexes a snapshot in one
transaction, so a failure or a stop leaves the index as it was:

1. **Extract** — stream files from git, extract the blobs whose chunks
   are not current, write the snapshot's file list.
2. **Edges** — infer the snapshot's edges, and again those of every
   indexed snapshot listing a blob extracted here.
3. **Finish** — mark the snapshot indexed, clean up orphans.

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

import dataframely as dy
import structlog

from rbtr.config import config
from rbtr.domain.edge_frames import EdgeChunkRow, InferredEdgeRow
from rbtr.domain.edges import infer_edges
from rbtr.domain.models import (
    FileOutcome,
    FileSnapshot,
    IndexResult,
    Repo,
)
from rbtr.errors import BuildStopped
from rbtr.git import (
    FileEntry,
    changed_files,
    list_files,
    non_commit_shas,
    normalise_repo_path,
    read_blob,
)
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

    Files outside *changed* keep their extension or stored language, a
    stored one only while it is installed; content detection applies to
    new or changed files. Plaintext is the fallback when no language
    claims a file.
    """
    mgr = get_manager()
    resolved: list[tuple[FileEntry, str]] = []
    for entry in files:
        installed = stored.get(entry.path, "")
        if mgr.get_registration(installed) is None:
            installed = ""
        language = mgr.detect_language(entry.path) or installed
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


def snapshot_edges(
    repo_path: str, chunks: dy.DataFrame[EdgeChunkRow], files: dict[str, str]
) -> dy.DataFrame[InferredEdgeRow]:
    """The edges a snapshot's *chunks* make, its files *files* (`{path: blob_sha}`).

    The manifests among *files* are read by blob from *repo_path*'s object
    store, which every checkout of the project shares; a blob git no
    longer holds is read as no manifest.
    """
    registrations = get_manager().registrations()
    names = {registration.manifest for registration in registrations if registration.manifest}
    manifests = {
        path: text.decode(errors="replace")
        for path, blob_sha in files.items()
        if path in names and (text := read_blob(repo_path, blob_sha)) is not None
    }
    resolutions = language_resolutions(registrations, manifests=manifests)
    return infer_edges(chunks, set(files), resolutions, steps=config.follow_steps)


def build_index(
    repo_path: str,
    snapshot_sha: str,
    index: Index,
    *,
    base_sha: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
    stop: threading.Event | None = None,
) -> IndexResult:
    """Build (or incrementally update) the index for *snapshot_sha*, in one transaction.

    Lists all files at *snapshot_sha*, extracts the blobs whose chunks
    are not current, infers the snapshot's edges, and again those of
    every indexed snapshot listing a blob extracted here, then marks the
    commit indexed and cleans up.  The commit becomes queryable via
    FTS/name/edges immediately — embedding is handled separately by
    `embed_index`.

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
    file, or before inferring the next snapshot's edges. The index is
    left as it was, as a build that died there would leave it, and the
    next build of it starts again.
    """
    t0 = time.monotonic()

    # Register before writing anything that refers to the repo, so a
    # crash part-way through leaves an unused repos row rather than
    # indexed rows that nothing can find.
    repo_path = normalise_repo_path(repo_path)
    repo = index.register(repo_path)

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
    # The `(blob_sha, file_language)` extractions this build writes: the
    # indexed snapshots listing one of them have their edges re-inferred.
    extracted: set[tuple[str, str]] = set()

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
                extracted.update((entry.blob_sha, language) for entry, language in to_extract)

            for listed_file in batch:
                if stop is not None and stop.is_set():
                    raise BuildStopped
                entry = listed_file.entry
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
        # The snapshot's edges, and those of every indexed snapshot listing a
        # blob extracted here, which reads its new chunks (ARCHITECTURE, "A
        # blob's chunks and the edges read from them commit together").
        built = repo.at(snapshot_sha)
        inferred = list(
            dict.fromkeys([built, *session.snapshots_listing(extracted, repos=index.repos())])
        )
        for done, at in enumerate(inferred):
            if stop is not None and stop.is_set():
                raise BuildStopped
            on_progress("edges", done, len(inferred))
            listed = session.snapshot_files(at=at)
            located = session.edge_chunks(at=at)
            edges = snapshot_edges(repo_path, located, listed)
            session.replace_edges(edges, at=at)
            log.info(
                "inferred_edges", repo=at.repo.repo_path, sha=at.snapshot_sha[:12], edges=len(edges)
            )
            if at == built:
                result.stats.total_edges, result.stats.total_chunks = len(edges), len(located)
        on_progress("finalising", 0, 0)
        session.mark_indexed(snapshot_sha=snapshot_sha)
        cleaned = session.cleanup()
        if cleaned.file_snapshots or cleaned.edges or cleaned.chunks:
            log.info(
                "cleanup",
                file_snapshots=cleaned.file_snapshots,
                edges=cleaned.edges,
                chunks=cleaned.chunks,
            )

    # Every outcome that occurred, by name, so a run's file count is
    # accounted for in the log line that reports it.
    log.info(
        "extracted_files",
        total=result.stats.total_files,
        sha=snapshot_sha[:12],
        **{outcome.value: count for outcome, count in result.stats.outcomes.items()},
    )
    result.stats.elapsed_seconds = time.monotonic() - t0
    log.info(
        "index_complete",
        chunks=result.stats.total_chunks,
        edges=result.stats.total_edges,
        elapsed_seconds=round(result.stats.elapsed_seconds, 1),
    )
    return result
