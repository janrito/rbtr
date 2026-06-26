"""Index orchestration — build and diff a code index.

`build_index` is the entry point.  It runs four phases, each
in its own transactional session:

1. **Extract** — stream files from git, extract chunks via
   tree-sitter or plugin chunkers, write chunks + snapshots.
2. **Edges** — infer import/test/doc edges from the committed
   chunk set.
3. **Embed** — compute embeddings for chunks that lack them.
4. **Finalise** — mark the commit indexed, clean up orphans.

All heavy work runs synchronously — the caller (daemon job
worker) runs it via `asyncio.to_thread()`.

Progress is reported via a single `ProgressCallback(phase,
done, total)`.  Logs record completion summaries; progress
reports real-time state.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import structlog

from rbtr.config import config
from rbtr.git import FileEntry, changed_files, list_files
from rbtr.index.chunks import chunk_plaintext, detect_prose_format
from rbtr.index.edges import build_resolution_map, infer_import_edges, infer_test_edges
from rbtr.index.embeddings import Embedder, embedding_text
from rbtr.index.models import (
    Chunk,
    Edge,
    IndexResult,
    Snapshot,
    TokenisedChunk,
)
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from rbtr.index.treesitter import extract_symbols
from rbtr.languages import get_manager
from rbtr.languages.hookspec import LanguageRegistration, build_import_from_captures
from rbtr.logging import elapsed_ms
from rbtr.rbtrignore import load_ignore

log = structlog.get_logger(__name__)

type ProgressCallback = Callable[[str, int, int], None]
"""`(phase, done, total)` — called to report build progress."""


def _noop_progress(_phase: str, _done: int, _total: int) -> None:
    pass


# ── File extraction ─────────────────────────────────────────────────


def _extract_file(
    entry: FileEntry, language: str, reg: LanguageRegistration | None
) -> Iterator[Chunk]:
    """Route a single file to the appropriate extraction strategy.

    The caller resolves the language and registration; this
    function only routes to the correct chunker.

    Yields chunks without materialising the full list.  The
    caller is responsible for blob dedup (`has_blob`) and
    deleting stale chunks before calling this.
    """
    mgr = get_manager()

    if reg is not None and reg.chunker is not None:
        grammar = mgr.load_grammar(language)
        if grammar is not None:
            for c in reg.chunker(
                entry.path,
                entry.blob_sha,
                entry.content.decode(errors="replace"),
                grammar,
            ):
                c.language = language
                yield c
            return

    if language:
        grammar = mgr.load_grammar(language)
        query_str = mgr.get_query(language)
        if grammar is not None and query_str is not None:
            yield from extract_symbols(
                entry.path,
                entry.blob_sha,
                entry.content,
                grammar,
                query_str,
                language=language,
                import_extractor=(
                    reg.import_extractor
                    if reg and reg.import_extractor
                    else build_import_from_captures
                ),
                scope_types=reg.scope_types if reg else frozenset(),
                doc_comment_node_types=(reg.doc_comment_node_types if reg else frozenset()),
            )
            return

    # Neither detected nor prose — raw line-based chunks.
    text = entry.content.decode(errors="replace")
    yield from chunk_plaintext(entry.path, entry.blob_sha, text)


# ── Build phases ─────────────────────────────────────────────────────


def _extract_and_store_chunks(
    *,
    store: IndexStore,
    repo_path: str,
    commit_sha: str,
    repo_id: int,
    base_sha: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
) -> tuple[IndexResult, set[str]]:
    """Stream files, extract chunks, write snapshots.

    When *base_sha* is provided, only files that changed between
    *base_sha* and *commit_sha* are extracted.

    Opens its own session with an explicit sweep.
    """
    mgr = get_manager()
    repo_root = Path(repo_path).resolve()
    ignore = load_ignore(repo_root)
    changed: set[str] | None = None
    if base_sha is not None:
        changed = changed_files(repo_path, base_sha, commit_sha)

    snapshots: list[Snapshot] = []
    result = IndexResult()
    repo_files: set[str] = set()  # collected for edge inference

    with store.session() as session:
        session.sweep()
        # Stream files from git.  Content is only held during
        # extraction, then released by the iterator.
        for entry in list_files(
            repo_path,
            commit_sha,
            max_file_size=config.max_file_size,
            ignore=ignore,
        ):
            result.stats.total_files += 1
            repo_files.add(entry.path)

            # In incremental mode, skip files that didn't change.
            if changed is not None and entry.path not in changed:
                result.stats.skipped_files += 1
                snapshots.append(
                    Snapshot(
                        commit_sha=commit_sha,
                        file_path=entry.path,
                        blob_sha=entry.blob_sha,
                    )
                )
                continue

            # Resolve language: extension first, then stored
            # detection, then content sniff (new files only).
            detected_lang = mgr.detect_language(entry.path) or ""
            if not detected_lang:
                detected_lang = store.get_snapshot_language(entry.path, repo_id=repo_id)
            if not detected_lang:
                text = entry.content.decode(errors="replace")
                fmt = detect_prose_format(text)
                if fmt:
                    detected_lang = fmt

            # Resolve version from registration.
            reg = mgr.get_registration(detected_lang) if detected_lang else None
            version = reg.language_plugin_version if reg else 1

            # Blob dedup gate.
            if store.has_blob(
                entry.blob_sha,
                language=detected_lang,
                language_plugin_version=version,
            ):
                result.stats.skipped_files += 1
            else:
                try:
                    # Delete old chunks before re-extraction (language may
                    # have changed, producing different chunk IDs).
                    session.delete_chunks_for_blobs({entry.blob_sha})
                    file_has_chunks = False
                    for chunk in _extract_file(entry, detected_lang, reg):
                        tokenised = TokenisedChunk(
                            **chunk.model_dump(),
                            content_tokens=tokenise_code(chunk.content),
                            name_tokens=tokenise_code(chunk.name),
                            language_plugin_version=version,
                        )
                        session.add_chunk(tokenised)
                        file_has_chunks = True
                    if file_has_chunks:
                        result.stats.parsed_files += 1
                except Exception:
                    msg = f"Failed to index {entry.path}"
                    log.exception("index_file_failed", path=entry.path)
                    result.errors.append(msg)

            # Record detected language on snapshot.
            snapshots.append(
                Snapshot(
                    commit_sha=commit_sha,
                    file_path=entry.path,
                    blob_sha=entry.blob_sha,
                    detected_language=detected_lang,
                )
            )

            on_progress("parsing", result.stats.total_files, result.stats.total_files)

        session.replace_snapshots(commit_sha, snapshots, repo_id=repo_id)

    log.info(
        "extracted_files",
        total=result.stats.total_files,
        parsed=result.stats.parsed_files,
        skipped=result.stats.skipped_files,
        sha=commit_sha[:12],
    )
    return result, repo_files


def _infer_and_store_edges(
    *,
    store: IndexStore,
    chunks: list[Chunk],
    repo_files: set[str],
    commit_sha: str,
    repo_id: int,
    on_progress: ProgressCallback,
) -> int:
    """Infer cross-file edges and write them. Returns edge count."""
    on_progress("edges", 0, 0)
    mgr = get_manager()
    resolution_map = build_resolution_map(mgr)
    edges: list[Edge] = []
    edges.extend(infer_import_edges(chunks, repo_files, resolution_map))
    edges.extend(infer_test_edges(chunks, repo_files, resolution_map))

    with store.session() as session:
        session.replace_edges(commit_sha, edges, repo_id=repo_id)

    log.info("inferred_edges", edges=len(edges))
    return len(edges)


def _mark_indexed_and_cleanup(
    *, store: IndexStore, repo_id: int, commit_sha: str, on_progress: ProgressCallback
) -> None:
    """Mark the commit indexed and remove orphaned data."""
    on_progress("finalising", 0, 0)
    with store.session() as session:
        session.mark_indexed(repo_id, commit_sha)
        cleaned = session.cleanup(repo_id)
        if cleaned.snapshots or cleaned.edges or cleaned.chunks:
            log.info(
                "cleanup",
                snapshots=cleaned.snapshots,
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
            sha=commit_sha[:12],
        )


# ── Public API ───────────────────────────────────────────────────────


def build_index(
    repo_path: str,
    commit_sha: str,
    store: IndexStore,
    *,
    repo_id: int,
    base_sha: str | None = None,
    on_progress: ProgressCallback = _noop_progress,
) -> IndexResult:
    """Build (or incrementally update) the index for *commit_sha*.

    Lists all files at *commit_sha*, extracts chunks, infers
    edges, and marks the commit indexed.  The commit becomes
    queryable via FTS/name/edges immediately — embedding is
    handled separately by `embed_index`.

    When *base_sha* is provided, only files that changed between
    *base_sha* and *commit_sha* are considered for extraction.
    Unchanged files are skipped without checking `has_blob`.
    """
    t0 = time.monotonic()

    # Phase 1: extract chunks from git, write to DB.
    result, repo_files = _extract_and_store_chunks(
        store=store,
        repo_path=repo_path,
        commit_sha=commit_sha,
        repo_id=repo_id,
        base_sha=base_sha,
        on_progress=on_progress,
    )

    # Fetch committed chunks for edge inference.
    # Lightweight: skips content_tokens/name_tokens (~37% smaller).
    all_chunks = store.get_chunks(commit_sha, repo_id=repo_id)

    # Phase 2: infer cross-file edges.
    result.stats.total_edges = _infer_and_store_edges(
        store=store,
        chunks=all_chunks,
        repo_files=repo_files,
        commit_sha=commit_sha,
        repo_id=repo_id,
        on_progress=on_progress,
    )

    # Phase 3: mark complete and remove orphaned data.
    _mark_indexed_and_cleanup(
        store=store, repo_id=repo_id, commit_sha=commit_sha, on_progress=on_progress
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


def embed_index(
    store: IndexStore,
    commit_sha: str,
    *,
    repo_id: int,
    embedder: Embedder,
    on_progress: ProgressCallback = _noop_progress,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    """Embed un-embedded chunks for an already-indexed commit.

    Fetches unembedded chunks in pages and processes each page
    in batches.  Each batch gets its own write session so the
    DuckDB write lock is released between batches — higher-priority
    builds can run in the gaps.

    When *should_stop* returns ``True`` the function commits
    the current batch and returns early.  The remaining chunks
    are still ``embedding IS NULL`` so the next call picks up
    where this one left off.

    Returns the number of chunks that were embedded.
    """
    total = store.count_unembedded(repo_id, commit_sha)
    if total == 0:
        return 0

    on_progress("loading_model", 0, 0)

    done = 0
    t0 = time.perf_counter()

    while missing := store.get_unembedded_chunks(repo_id, commit_sha):
        before = done
        for batch in itertools.batched(missing, config.embedding_batch_size, strict=False):
            texts = [embedding_text(c.name, c.content) for c in batch]
            try:
                results = embedder.embed(texts)
            except (RuntimeError, ValueError):
                log.warning("embedding_batch_failed", exc_info=True)
                continue
            with store.session() as session:
                session.update_embeddings(
                    [c.id for c in batch],
                    [r.vector for r in results],
                    truncated=[r.truncated for r in results],
                )
            done += len(batch)
            on_progress("embedding", done, total)
            if should_stop is not None and should_stop():
                log.info("embedding_preempted", done=done, total=total)
                return done
        if done == before:
            break

    log.info("embedded_chunks", done=done, total=total, elapsed_ms=elapsed_ms(t0))
    return done
