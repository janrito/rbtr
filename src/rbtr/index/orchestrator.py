"""Index orchestration — build, update, and diff a code index.

Coordinates between the git reader, language plugins, tree-sitter
extraction, chunking, edge inference, embeddings, and DuckDB
storage.  All heavy work runs synchronously — the caller (engine)
is responsible for running it in a background thread.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pygit2

from rbtr.config import config
from rbtr.index.chunks import chunk_plaintext
from rbtr.index.edges import infer_doc_edges, infer_import_edges, infer_test_edges
from rbtr.index.git import FileEntry, list_files
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind, IndexStats
from rbtr.index.store import IndexStore
from rbtr.plugins.manager import get_manager

log = logging.getLogger(__name__)

# ── Types ────────────────────────────────────────────────────────────

ProgressCallback = Callable[[int, int], None]
"""``(files_done, total_files)`` — called after each file is processed."""


@dataclass
class IndexResult:
    """Outcome of an index build or update."""

    stats: IndexStats = field(default_factory=IndexStats)
    errors: list[str] = field(default_factory=list)


# ── File routing ─────────────────────────────────────────────────────


def _extract_file(entry: FileEntry) -> list[Chunk]:
    """Route a single file to the appropriate extraction strategy.

    1. Plugin with custom chunker → delegate to it.
    2. Language with grammar + query → tree-sitter symbol extraction.
    3. Everything else → plaintext line-based fallback.
    """
    path = entry.path
    text = entry.content.decode(errors="replace")

    mgr = get_manager()
    lang_id = mgr.detect_language(path)

    if lang_id is not None:
        reg = mgr.get_registration(lang_id)

        # Plugin-provided chunker (e.g. Markdown heading-hierarchy).
        if reg is not None and reg.chunker is not None:
            return reg.chunker(path, entry.blob_sha, text)

        grammar = mgr.load_grammar(lang_id)
        query_str = mgr.get_query(lang_id)

        if grammar is not None and query_str is not None:
            # Deferred import: treesitter pulls in the native
            # tree-sitter lib which is heavy at import time.
            from rbtr.index.treesitter import extract_symbols

            return extract_symbols(
                path,
                entry.blob_sha,
                entry.content,
                grammar,
                query_str,
                import_extractor=reg.import_extractor if reg else None,
                scope_types=reg.scope_types if reg else frozenset(),
            )

    # Fallback: plaintext line-based chunks.
    return chunk_plaintext(path, entry.blob_sha, text)


# ── Full index ───────────────────────────────────────────────────────


def build_index(
    repo: pygit2.Repository,
    commit_sha: str,
    store: IndexStore,
    *,
    on_progress: ProgressCallback | None = None,
    on_embed_progress: ProgressCallback | None = None,
) -> IndexResult:
    """Build a full index for *commit_sha*.

    Skips files whose ``blob_sha`` is already in the store (blob
    dedup).  After all files are processed, infers cross-file edges
    and embeds all un-embedded chunks.

    Parameters:
        repo:              Open pygit2 repository.
        commit_sha:        Git ref or SHA to index.
        store:             DuckDB store (already initialised).
        on_progress:       Optional callback ``(done, total)`` for file
                           extraction progress.
        on_embed_progress: Optional callback ``(done, total)`` for
                           embedding progress.
    """
    t0 = time.monotonic()
    result = IndexResult()

    # 1. List indexable files.
    files = list(list_files(repo, commit_sha))
    total = len(files)
    result.stats.total_files = total
    log.info("Indexing %d files at %s", total, commit_sha[:12])

    all_chunks: list[Chunk] = []
    snapshot_rows: list[tuple[str, str, str]] = []

    # 2. Per-file: extract chunks, record snapshots.
    for i, entry in enumerate(files):
        snapshot_rows.append((commit_sha, entry.path, entry.blob_sha))

        if store.has_blob(entry.blob_sha):
            result.stats.skipped_files += 1
        else:
            try:
                chunks = _extract_file(entry)
                if chunks:
                    all_chunks.extend(chunks)
                    result.stats.parsed_files += 1
            except Exception:
                msg = f"Failed to index {entry.path}"
                log.exception(msg)
                result.errors.append(msg)

        if on_progress is not None:
            on_progress(i + 1, total)

    # Batch insert all chunks and snapshots in one call each.
    store.insert_chunks(all_chunks)
    # Replace the snapshot set for this ref so deleted files from
    # older reviews don't leak into current queries.
    store.delete_snapshots(commit_sha)
    store.insert_snapshots(snapshot_rows)

    # We also need chunks from skipped (already stored) files for edge inference.
    if result.stats.skipped_files > 0:
        all_chunks = store.get_chunks(commit_sha)

    # 3. Cross-file edges (clear stale edges first for idempotency).
    store.delete_edges(commit_sha)
    repo_files = {entry.path for entry in files}
    edges: list[Edge] = []
    edges.extend(infer_import_edges(all_chunks, repo_files))
    edges.extend(infer_test_edges(all_chunks, repo_files))
    edges.extend(infer_doc_edges(all_chunks))
    store.insert_edges(edges, commit_sha)
    result.stats.total_edges = len(edges)

    # 4. Flush inserts so reads from the UI thread see chunks/edges
    #    while the (slow) embedding phase runs.
    store.checkpoint()

    # 5. Embeddings.
    _embed_missing(store, all_chunks, on_progress=on_embed_progress)

    # 6. FTS index is rebuilt lazily on first search_fulltext call
    #    via store._ensure_fts().  No explicit rebuild needed here —
    #    insert_chunks already sets _fts_dirty.

    # 7. Prune orphaned chunks/edges from previous runs.
    pruned_chunks, pruned_edges = store.prune_orphans()
    if pruned_chunks or pruned_edges:
        log.info("Pruned %d orphan chunks, %d orphan edges", pruned_chunks, pruned_edges)

    result.stats.total_chunks = len(all_chunks)
    result.stats.elapsed_seconds = time.monotonic() - t0
    log.info(
        "Index complete: %d chunks, %d edges, %.1fs",
        result.stats.total_chunks,
        result.stats.total_edges,
        result.stats.elapsed_seconds,
    )
    return result


# ── Incremental update ───────────────────────────────────────────────


def update_index(
    repo: pygit2.Repository,
    base_sha: str,
    head_sha: str,
    store: IndexStore,
    *,
    on_progress: ProgressCallback | None = None,
    on_embed_progress: ProgressCallback | None = None,
) -> IndexResult:
    """Incrementally index *head_sha* given an existing index at *base_sha*.

    Only re-extracts files that changed between the two commits.
    Copies unchanged snapshots from *base_sha*, then re-infers all
    edges for *head_sha* (edges are cheap to recompute).

    Parameters:
        repo:              Open pygit2 repository.
        base_sha:          Already-indexed commit (the base branch).
        head_sha:          New commit to index (the PR head).
        store:             DuckDB store with *base_sha* already indexed.
        on_progress:       Optional file extraction progress callback.
        on_embed_progress: Optional embedding progress callback.
    """
    from rbtr.index.git import changed_files

    t0 = time.monotonic()
    result = IndexResult()

    changed = changed_files(repo, base_sha, head_sha)
    log.info("Incremental index: %d changed files", len(changed))

    # List all files at head_sha for snapshots + edge inference.
    head_files = list(list_files(repo, head_sha))
    result.stats.total_files = len(head_files)

    all_chunks: list[Chunk] = []
    snapshot_rows: list[tuple[str, str, str]] = []
    changed_entries: list[FileEntry] = []

    for entry in head_files:
        snapshot_rows.append((head_sha, entry.path, entry.blob_sha))
        if entry.path in changed:
            changed_entries.append(entry)
        else:
            result.stats.skipped_files += 1

    # Replace snapshots for the entire head tree so deleted files
    # from previous runs at this ref do not linger.
    store.delete_snapshots(head_sha)
    store.insert_snapshots(snapshot_rows)

    # Extract only changed files.
    new_chunks: list[Chunk] = []
    total = len(changed_entries)
    for i, entry in enumerate(changed_entries):
        if store.has_blob(entry.blob_sha):
            result.stats.skipped_files += 1
        else:
            try:
                chunks = _extract_file(entry)
                if chunks:
                    new_chunks.extend(chunks)
                    result.stats.parsed_files += 1
            except Exception:
                msg = f"Failed to index {entry.path}"
                log.exception(msg)
                result.errors.append(msg)

        if on_progress is not None:
            on_progress(i + 1, total)

    # Batch insert new chunks.
    store.insert_chunks(new_chunks)

    # Fetch all chunks at head for edge inference.
    all_chunks = store.get_chunks(head_sha)

    # Re-infer edges for the head commit (clear stale edges first).
    store.delete_edges(head_sha)
    repo_files = {entry.path for entry in head_files}
    edges: list[Edge] = []
    edges.extend(infer_import_edges(all_chunks, repo_files))
    edges.extend(infer_test_edges(all_chunks, repo_files))
    edges.extend(infer_doc_edges(all_chunks))
    store.insert_edges(edges, head_sha)
    result.stats.total_edges = len(edges)

    # Flush inserts so reads from the UI thread see chunks/edges
    # while the (slow) embedding phase runs.
    store.checkpoint()

    # Embeddings for new chunks.
    _embed_missing(store, all_chunks, on_progress=on_embed_progress)

    # FTS index is rebuilt lazily on first search_fulltext call.

    # Prune orphaned chunks/edges from previous runs.
    pruned_chunks, pruned_edges = store.prune_orphans()
    if pruned_chunks or pruned_edges:
        log.info("Pruned %d orphan chunks, %d orphan edges", pruned_chunks, pruned_edges)

    result.stats.total_chunks = len(all_chunks)
    result.stats.elapsed_seconds = time.monotonic() - t0
    log.info(
        "Incremental index: %d chunks, %d edges, %.1fs",
        result.stats.total_chunks,
        result.stats.total_edges,
        result.stats.elapsed_seconds,
    )
    return result


# ── Semantic diff ────────────────────────────────────────────────────


@dataclass
class SemanticDiff:
    """Structural differences between two indexed commits."""

    added: list[Chunk] = field(default_factory=list)
    """Symbols that exist in head but not in base."""

    removed: list[Chunk] = field(default_factory=list)
    """Symbols that exist in base but not in head."""

    modified: list[Chunk] = field(default_factory=list)
    """Symbols at the same path whose content changed."""

    stale_docs: list[tuple[Chunk, Chunk]] = field(default_factory=list)
    """``(doc_chunk, code_chunk)`` where the code changed but
    the doc referencing it did not."""

    missing_tests: list[Chunk] = field(default_factory=list)
    """New functions/methods with no ``TESTS`` edge."""

    broken_edges: list[Edge] = field(default_factory=list)
    """Import edges in head that pointed at symbols now removed."""


def compute_diff(
    base_sha: str,
    head_sha: str,
    store: IndexStore,
) -> SemanticDiff:
    """Compute structural differences between two indexed commits.

    Both commits must already be indexed in *store*.

    Uses file-level diff from DuckDB (added/removed/modified files),
    then performs symbol-level comparison within modified files to
    surface individual added, removed, and changed symbols.
    """
    diff = SemanticDiff()

    symbol_kinds = frozenset({ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD})

    # 1. File-level diff from DuckDB.
    added_raw, removed_raw, modified_raw = store.diff_chunks(base_sha, head_sha)

    # Symbols in entirely new files → added.
    diff.added = [c for c in added_raw if c.kind in symbol_kinds]

    # Symbols in entirely removed files → removed.
    diff.removed = [c for c in removed_raw if c.kind in symbol_kinds]

    # 2. Symbol-level diff within modified files.
    #    modified_raw contains head-side chunks of files that changed.
    #    We need base-side chunks too to compare.
    modified_files = {c.file_path for c in modified_raw}
    base_chunks = store.get_chunks(base_sha)
    base_by_key: dict[tuple[str, str, str], Chunk] = {}
    for c in base_chunks:
        if c.file_path in modified_files and c.kind in symbol_kinds:
            base_by_key[(c.file_path, c.name, c.scope)] = c

    head_by_key: dict[tuple[str, str, str], Chunk] = {}
    for c in modified_raw:
        if c.kind in symbol_kinds:
            head_by_key[(c.file_path, c.name, c.scope)] = c

    for key, head_chunk in head_by_key.items():
        base_chunk = base_by_key.get(key)
        if base_chunk is None:
            diff.added.append(head_chunk)
        elif base_chunk.content != head_chunk.content:
            diff.modified.append(head_chunk)

    for key, base_chunk in base_by_key.items():
        if key not in head_by_key:
            diff.removed.append(base_chunk)

    removed_ids = {c.id for c in diff.removed}

    # 3. Stale docs: doc edges that point at modified code but
    #    the doc chunk itself is unchanged.
    head_edges = store.get_edges(head_sha)
    head_chunks_by_id: dict[str, Chunk] = {}
    for c in store.get_chunks(head_sha):
        head_chunks_by_id[c.id] = c

    modified_code_files = {c.file_path for c in diff.modified}
    doc_chunk_ids_in_modified = set()
    for c in modified_raw:
        if c.kind == ChunkKind.DOC_SECTION:
            doc_chunk_ids_in_modified.add(c.id)

    for edge in head_edges:
        if edge.kind != EdgeKind.DOCUMENTS:
            continue
        doc = head_chunks_by_id.get(edge.source_id)
        code = head_chunks_by_id.get(edge.target_id)
        if doc is None or code is None:
            continue
        # Code was modified but doc was not.
        if code.file_path in modified_code_files and doc.id not in doc_chunk_ids_in_modified:
            diff.stale_docs.append((doc, code))

    # 4. Missing tests: new functions/methods with no TESTS edge.
    #    Exclude test functions themselves — they don't need tests.
    test_target_ids = {e.target_id for e in head_edges if e.kind == EdgeKind.TESTS}
    for chunk in diff.added:
        if chunk.kind not in (ChunkKind.FUNCTION, ChunkKind.METHOD):
            continue
        if chunk.id in test_target_ids:
            continue
        if chunk.kind == ChunkKind.TEST_FUNCTION:
            continue
        if chunk.name.startswith("test_") or "/test" in chunk.file_path:
            continue
        diff.missing_tests.append(chunk)

    # 5. Broken edges: head import edges whose target was removed.
    for edge in head_edges:
        if edge.kind == EdgeKind.IMPORTS and edge.target_id in removed_ids:
            diff.broken_edges.append(edge)

    return diff


# ── Embedding helper ─────────────────────────────────────────────────


def _embed_missing(
    store: IndexStore,
    chunks: list[Chunk],
    *,
    on_progress: ProgressCallback | None = None,
) -> None:
    """Embed chunks that don't have an embedding yet.

    Embedding is best-effort: if the model cannot be loaded (missing
    GGUF, llama-cpp not installed, GPU init failure, etc.) the error
    is logged and the structural index remains usable — only
    ``search_similar`` will be degraded.
    """
    missing = [c for c in chunks if not c.embedding]
    if not missing:
        return

    try:
        from rbtr.index.embeddings import embed_texts  # deferred: heavy native lib
    except Exception:
        log.warning("Embedding model unavailable — skipping embeddings", exc_info=True)
        return

    batch_size = config.index.embedding_batch_size
    total = len(missing)
    log.info("Embedding %d chunks", total)
    done = 0
    for i in range(0, total, batch_size):
        batch = missing[i : i + batch_size]
        texts = [f"{c.name}\n{c.content}" for c in batch]
        try:
            vectors = embed_texts(texts)
        except Exception:
            log.warning("Embedding batch %d failed — skipping", i, exc_info=True)
            done += len(batch)
            if on_progress is not None:
                on_progress(done, total)
            continue
        store.update_embeddings([c.id for c in batch], vectors)
        done += len(batch)
        if on_progress is not None:
            on_progress(done, total)
