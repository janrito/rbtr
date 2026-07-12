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
from typing import TYPE_CHECKING

import structlog
from tree_sitter import Parser, QueryCursor

from rbtr.config import config
from rbtr.git import FileEntry, changed_files, list_files
from rbtr.index.chunks import chunk_plaintext, detect_prose_format, host_presence_chunk
from rbtr.index.edges import build_resolution_map, infer_import_edges
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
from rbtr.index.treesitter import _get_query, extract_symbols
from rbtr.languages.manager import LanguageManager, get_manager
from rbtr.languages.registration import ChunkExtraction, QueryExtraction
from rbtr.logging import elapsed_ms
from rbtr.rbtrignore import load_ignore

if TYPE_CHECKING:
    from tree_sitter import Node, Range

log = structlog.get_logger(__name__)

type ProgressCallback = Callable[[str, int, int], None]
"""`(phase, done, total)` — called to report build progress."""


def _noop_progress(_phase: str, _done: int, _total: int) -> None:
    pass


# ── File extraction ─────────────────────────────────────────────────


def extract_query(
    language: str,
    file_path: str,
    blob_sha: str,
    content: bytes,
    ranges: list[Range] | None = None,
    doc_comment_node_types: frozenset[str] | None = None,
) -> Iterator[Chunk]:
    """Run *language*'s tree-sitter query over *content*.

    The single query-path extraction entry. `ranges=None` parses the
    whole file; a range list restricts parsing to those spans (an
    embedded block, e.g. an SFC `<script>`), reporting absolute
    positions. Yields nothing if *language* has no grammar or query.
    The injection runner (`extract_injections`) calls this to delegate
    each embedded block to its language.

    *doc_comment_node_types* defaults to the registration's; pass an
    explicit (possibly empty) set to override leading-comment attachment.
    """
    mgr = get_manager()
    reg = mgr.get_registration(language)
    grammar = mgr.load_grammar(language)
    if reg is None or not isinstance(reg.extraction, QueryExtraction) or grammar is None:
        return
    yield from extract_symbols(
        reg,
        file_path,
        blob_sha,
        content,
        grammar,
        doc_comment_node_types=doc_comment_node_types,
        included_ranges=ranges,
    )


def extract_primary(
    language: str,
    file_path: str,
    blob_sha: str,
    content: bytes,
    ranges: list[Range] | None = None,
) -> list[Chunk] | None:
    """Run *language*'s primary extraction (its chunker or query) over *content*.

    Restricted to *ranges* when given — an embedded block — so it doubles as
    the injection delegate: the full target plugin runs on the range. Every
    returned chunk carries *language* (query extraction sets it already; a
    chunker's blank chunks are filled here), so a delegated chunker target is
    labelled correctly. Returns None when the language has neither a chunker
    nor a query, leaving the caller to fall back to plaintext.
    """
    mgr = get_manager()
    grammar = mgr.load_grammar(language)
    reg = mgr.get_registration(language)
    extraction = reg.extraction if reg is not None else None
    if isinstance(extraction, ChunkExtraction) and grammar is not None:
        chunks = list(
            extraction.chunker(
                file_path, blob_sha, content.decode(errors="replace"), grammar, ranges
            )
        )
    elif isinstance(extraction, QueryExtraction) and grammar is not None:
        chunks = list(extract_query(language, file_path, blob_sha, content, ranges=ranges))
    else:
        return None
    for chunk in chunks:
        if not chunk.language:
            chunk.language = language
    return chunks


def _resolve_injection_hint(mgr: LanguageManager, captures: dict[str, list[Node]]) -> str | None:
    """Resolve a dynamic `@injection.language` hint to a language id.

    A markdown fence tags its block with a free-form name (` ```python `,
    ` ```py `). Treat the name as a language id, then as a file extension, so
    both spellings reach the python plugin. An unknown hint returns None and
    the block is left unparsed.
    """
    nodes = captures.get("injection.language")
    if not nodes or nodes[0].text is None:
        return None
    hint = nodes[0].text.decode().strip().lower()
    return hint if mgr.get_registration(hint) else mgr.detect_language(f"x.{hint}")


_MAX_INJECTION_DEPTH = 5
"""Recursion cap for nested injection (a host embedded in a host). Insurance
only — delegation terminates naturally as ranges shrink and a target with no
`injection_query` stops."""


def extract_injections(
    language: str,
    file_path: str,
    blob_sha: str,
    content: bytes,
    ranges: list[Range] | None = None,
    _depth: int = 0,
) -> Iterator[Chunk]:
    """Yield chunks for code embedded in *content* via *language*'s injections.

    A host language declares an `injection_query` marking each embedded block
    and the language to parse it as; each block's range is delegated to that
    language's *full* primary extraction (`extract_primary` — chunker or
    query), so a `<script lang="ts">` block yields TypeScript chunks and a
    fenced ` ```html ` block yields HTML chunks, both at real line numbers.
    The target is a static `#set! injection.language`, or — for a free-form
    hint like a markdown fence — a captured `@injection.language` resolved to
    a language id. A block matching several rules uses the highest priority.

    Delegation recurses: the target's own injection runs on the block too, so
    an HTML block containing an inline `<script>` also yields its js, bounded
    by `_MAX_INJECTION_DEPTH`. *ranges* restricts the host parse to a block
    (used by that recursion); None parses the whole file.
    """
    if _depth > _MAX_INJECTION_DEPTH:
        return
    mgr = get_manager()
    reg = mgr.get_registration(language)
    if reg is None or reg.injection_query is None:
        return
    grammar = mgr.load_grammar(language)
    if grammar is None:
        return

    query = _get_query(grammar, reg.injection_query)
    parser = Parser(grammar)
    if ranges is not None:
        parser.included_ranges = ranges
    tree = parser.parse(content)

    winner: dict[tuple[int, int], tuple[int, str, Range]] = {}
    for pattern, captures in QueryCursor(query).matches(tree.root_node):
        settings = query.pattern_settings(pattern)
        target = settings.get("injection.language") or _resolve_injection_hint(mgr, captures)
        if target is None:
            continue
        priority = int(settings.get("injection.priority") or "0")
        for block in captures.get("injection.content", []):
            if block.text is None or not block.text.strip():
                continue
            span = (block.start_byte, block.end_byte)
            if span not in winner or priority > winner[span][0]:
                winner[span] = (priority, target, block.range)

    for _priority, target, block_range in winner.values():
        delegated = extract_primary(target, file_path, blob_sha, content, ranges=[block_range])
        if delegated is not None:
            yield from delegated
        yield from extract_injections(
            target, file_path, blob_sha, content, ranges=[block_range], _depth=_depth + 1
        )


def extract_file(entry: FileEntry, language: str) -> list[Chunk]:
    """Extract a file's chunks, always including one in its own language.

    The per-file extraction entry point: the indexer calls this for every
    file, and it is public so plugin tests (and third-party language
    packages) drive the *real* pipeline rather than a copy. Given a file's
    content and language it picks the primary strategy — a registered
    chunker, the tree-sitter query, or plaintext line chunks — then adds
    any embedded-language injections (an SFC's `<script>`/`<style>`) on
    top. If none of that produced a chunk in the file's own *language*, a
    content-less host-presence chunk is appended so the dedup gate can skip
    the file on later builds instead of re-parsing it. The caller handles
    blob dedup and deletes stale chunks first.
    """
    text = entry.content.decode(errors="replace")
    reg = get_manager().get_registration(language)
    has_injection = reg is not None and reg.injection_query is not None

    primary = extract_primary(language, entry.path, entry.blob_sha, entry.content)
    if primary is not None:
        chunks = primary
    elif has_injection:
        chunks = []
    else:
        chunks = list(chunk_plaintext(entry.path, entry.blob_sha, text))

    if has_injection:
        chunks += extract_injections(language, entry.path, entry.blob_sha, entry.content)

    if not any(chunk.language == language for chunk in chunks):
        chunks.append(host_presence_chunk(entry.path, entry.blob_sha, language))

    return chunks


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
    # Each chunk is stamped with its OWN language's plugin version, not the
    # host file's. They coincide for single-language files; they differ for
    # multi-language files (SFCs), whose delegated chunks carry an embedded
    # language. Built once from the registry.
    version_by_language = {
        lang: reg.language_plugin_version
        for lang in mgr.all_language_ids()
        if (reg := mgr.get_registration(lang)) is not None
    }
    # The dedup gate checks every stored chunk against its language's current
    # version; `""` is the plaintext pseudo-language (always version 1).
    dedup_versions = {**version_by_language, "": 1}
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
            if store.has_blob(entry.blob_sha, detected_lang, dedup_versions):
                result.stats.skipped_files += 1
            else:
                try:
                    # Delete old chunks before re-extraction (language may
                    # have changed, producing different chunk IDs).
                    session.delete_chunks_for_blobs({entry.blob_sha})
                    file_has_chunks = False
                    for chunk in extract_file(entry, detected_lang):
                        tokenised = TokenisedChunk(
                            **chunk.model_dump(),
                            content_tokens=tokenise_code(chunk.content),
                            name_tokens=tokenise_code(chunk.name),
                            language_plugin_version=version_by_language.get(
                                chunk.language, version
                            ),
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
