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

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import structlog

from rbtr.config import config
from rbtr.domain.models import (
    Chunk,
    Edge,
    FileOutcome,
    FileSnapshot,
    IndexResult,
)
from rbtr.domain.tokenise import tokenise_code
from rbtr.git import changed_files, list_files, normalise_repo_path
from rbtr.index.progress import ProgressCallback, _noop_progress
from rbtr.index.staging import TokenisedChunk
from rbtr.index.store import IndexStore
from rbtr.languages.chunks import detect_prose_format, host_presence_chunk
from rbtr.languages.edges import build_resolution_map, infer_import_edges
from rbtr.languages.extract import extract_file
from rbtr.languages.manager import get_manager
from rbtr.languages.plaintext import PLAINTEXT
from rbtr.rbtrignore import load_ignore

log = structlog.get_logger(__name__)


# ── Build phases ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractedFile:
    """A file's extraction output, bundled with the language it used.

    The language a file was extracted as has to reach two places that
    must agree: its chunks' `file_language`, which is part of their
    identity, and its snapshot row's `detected_language`, which is how a
    read pairs the two. Holding one value and deriving both from it is
    what stops them disagreeing — a file recorded under one language
    with chunks stamped for another is indexed and unreachable.

    A skipped file is this with no chunks: it still needs its row.
    """

    path: str
    blob_sha: str
    language: str
    chunks: list[Chunk]

    def snapshot_row(self, snapshot_sha: str) -> FileSnapshot:
        """This file's row in *snapshot_sha*'s tree."""
        return FileSnapshot(
            snapshot_sha=snapshot_sha,
            file_path=self.path,
            blob_sha=self.blob_sha,
            detected_language=self.language,
        )

    def tokenised(self, serials: Mapping[str, int], default: int) -> Iterator[TokenisedChunk]:
        """The chunks as storage rows: FTS tokens derived, serial resolved.

        The single place tokenisation runs, so a chunk's tokens cannot
        disagree with the text they index. A chunk's serial comes from
        *its own* language, not the file's, so bumping an embedded
        grammar invalidates the blocks it parsed.
        """
        for chunk in self.chunks:
            yield TokenisedChunk(
                **chunk.model_dump(),
                content_tokens=tokenise_code(chunk.content),
                name_tokens=tokenise_code(chunk.name),
                extraction_serial=serials.get(chunk.language, default),
            )


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
    store: IndexStore,
    repo_path: str,
    snapshot_sha: str,
    repo_id: int,
    base_sha: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
) -> _Extracted:
    """Stream files, extract chunks, write snapshots.

    When *base_sha* is provided, only files that changed between
    *base_sha* and *snapshot_sha* are extracted.

    Opens its own session with an explicit sweep.
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
    # The dedup gate checks every stored chunk against its language's
    # current serial.  Plaintext is in the registry like any other, so
    # there is no pseudo-language to add here.
    dedup_serials = serial_by_language
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

    with store.session() as session:
        session.sweep()
        # Stream files from git.  Content is only held during
        # extraction, then released by the iterator.
        for entry in list_files(
            repo_path,
            snapshot_sha,
            max_file_size=config.max_file_size,
            ignore=ignore,
        ):
            repo_files.add(entry.path)
            if entry.path in manifest_names:
                manifests[entry.path] = entry.content.decode(errors="replace")

            # Resolve language: extension first, then stored detection.
            # Every file needs it, including one that is skipped below:
            # a snapshot row pairs with its chunks on the language the
            # file was extracted as, so a row without one has no chunks.
            detected_lang = mgr.detect_language(entry.path) or ""
            if not detected_lang:
                detected_lang = store.get_snapshot_language(entry.path, repo_id=repo_id)

            # A file contributes no chunks when it is unchanged since the
            # base snapshot, or when the gate finds its blob already
            # extracted this way. It still gets a row below.
            chunks: list[Chunk] = []
            serial = 1
            if changed is not None and entry.path not in changed:
                result.stats.record(FileOutcome.SKIPPED_UNCHANGED)
                detected_lang = detected_lang or PLAINTEXT
            else:
                # Content sniff, for a new file whose extension says nothing.
                if not detected_lang:
                    text = entry.content.decode(errors="replace")
                    fmt = detect_prose_format(text)
                    if fmt:
                        detected_lang = fmt
                # Nothing claimed it, so it is plaintext: a language, not
                # a blank. Settled before the gate, which asks whether
                # this blob was already extracted as this language.
                detected_lang = detected_lang or PLAINTEXT

                # Resolve the serial from registration.
                reg = mgr.get_registration(detected_lang)
                serial = reg.extraction_serial if reg else 1

                # Blob dedup gate.
                if store.blob_is_current(entry.blob_sha, detected_lang, dedup_serials):
                    result.stats.record(FileOutcome.SKIPPED_CURRENT)
                else:
                    try:
                        # Delete old chunks before re-extraction (language may
                        # have changed, producing different chunk IDs).
                        session.delete_chunks_for_blobs({entry.blob_sha}, detected_lang)
                        chunks = extract_file(entry, detected_lang)
                        result.stats.record(
                            FileOutcome.PARSED if chunks else FileOutcome.EXTRACTED_EMPTY
                        )
                    except Exception:
                        msg = f"Failed to index {entry.path}"
                        log.exception("index_file_failed", path=entry.path)
                        result.errors.append(msg)
                        result.stats.record(FileOutcome.FAILED)
                        # The row below records the file either way, so
                        # without this it is indexed but unreachable.
                        chunks = [host_presence_chunk(entry.path, entry.blob_sha, detected_lang)]

            extracted = ExtractedFile(entry.path, entry.blob_sha, detected_lang, chunks)
            for tokenised in extracted.tokenised(serial_by_language, serial):
                session.add_chunk(tokenised)
            snapshots.append(extracted.snapshot_row(snapshot_sha))

            on_progress("parsing", result.stats.total_files, result.stats.total_files)

        session.replace_snapshots(snapshot_sha, snapshots, repo_id=repo_id)

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
    store: IndexStore,
    chunks: list[Chunk],
    repo_files: set[str],
    manifests: dict[str, str],
    snapshot_sha: str,
    repo_id: int,
    on_progress: ProgressCallback,
) -> int:
    """Infer cross-file edges and write them. Returns edge count."""
    on_progress("edges", 0, 0)
    mgr = get_manager()
    resolution_map = build_resolution_map(mgr, manifests=manifests)
    edges: list[Edge] = []
    edges.extend(infer_import_edges(chunks, repo_files, resolution_map))

    with store.session() as session:
        session.replace_edges(snapshot_sha, edges, repo_id=repo_id)

    log.info("inferred_edges", edges=len(edges))
    return len(edges)


def _mark_indexed_and_cleanup(
    *, store: IndexStore, repo_id: int, snapshot_sha: str, on_progress: ProgressCallback
) -> None:
    """Mark the commit indexed and remove orphaned data."""
    on_progress("finalising", 0, 0)
    with store.session() as session:
        session.mark_indexed(repo_id, snapshot_sha)
        cleaned = session.cleanup(repo_id)
        if cleaned.file_snapshots or cleaned.edges or cleaned.chunks:
            log.info(
                "cleanup",
                file_snapshots=cleaned.file_snapshots,
                edges=cleaned.edges,
                chunks=cleaned.chunks,
            )
    # Invariant guard for the content-addressed store: a build commits a
    # commit's chunks and snapshots in one transaction, and cleanup has
    # just pruned unreferenced rows, so no chunk should now lack a
    # snapshot. A non-zero count means chunk and snapshot writes were
    # split across transactions somewhere — which would let another
    # repo's *global* orphan sweep delete chunks this repo still needs.
    # Warn (don't abort) so the condition is visible without breaking
    # indexing.
    orphans = store.count_orphan_chunks()
    if orphans:
        log.warning(
            "orphan_chunks_after_build",
            orphans=orphans,
            repo_id=repo_id,
            sha=snapshot_sha[:12],
        )


# ── Public API ───────────────────────────────────────────────────────


def build_index(
    repo_path: str,
    snapshot_sha: str,
    store: IndexStore,
    *,
    base_sha: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
) -> IndexResult:
    """Build (or incrementally update) the index for *snapshot_sha*.

    Lists all files at *snapshot_sha*, extracts chunks, infers
    edges, and marks the commit indexed.  The commit becomes
    queryable via FTS/name/edges immediately — embedding is
    handled separately by `embed_index`.

    *repo_path* may be any path inside the repository; it is resolved
    to the root and registered if not already known, since indexing a
    path is what puts a repo in the index.  The `repo_id` comes from
    that lookup rather than from the caller, so it always matches the
    path being built.  A linked worktree resolves to its own
    directory, and so is built as a repo of its own.

    When *base_sha* is provided, only files that changed between
    *base_sha* and *snapshot_sha* are considered for extraction.
    Unchanged files are skipped without checking `blob_is_current`.
    """
    t0 = time.monotonic()

    # Register before writing anything that refers to the repo, so a
    # crash part-way through leaves an unused repos row rather than
    # indexed rows that nothing can find.
    repo_path = normalise_repo_path(repo_path)
    with store.session() as session:
        repo_id = session.register_repo(repo_path)

    # Phase 1: extract chunks from git, write to DB.
    extracted = _extract_and_store_chunks(
        store=store,
        repo_path=repo_path,
        snapshot_sha=snapshot_sha,
        repo_id=repo_id,
        base_sha=base_sha,
        on_progress=on_progress,
    )

    # Fetch committed chunks for edge inference.
    # Lightweight: skips content_tokens/name_tokens (~37% smaller).
    result = extracted.result
    all_chunks = store.get_chunks(snapshot_sha, repo_id=repo_id)

    # Phase 2: infer cross-file edges.
    result.stats.total_edges = _infer_and_store_edges(
        store=store,
        chunks=all_chunks,
        repo_files=extracted.repo_files,
        manifests=extracted.manifests,
        snapshot_sha=snapshot_sha,
        repo_id=repo_id,
        on_progress=on_progress,
    )

    # Phase 3: mark complete and remove orphaned data.
    _mark_indexed_and_cleanup(
        store=store, repo_id=repo_id, snapshot_sha=snapshot_sha, on_progress=on_progress
    )

    result.stats.total_chunks = len(all_chunks)
    result.stats.elapsed_seconds = time.monotonic() - t0
    log.info(
        "index_complete",
        chunks=result.stats.total_chunks,
        edges=result.stats.total_edges,
        elapsed_seconds=round(result.stats.elapsed_seconds, 1),
    )
    return result
