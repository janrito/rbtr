"""Handler for /index — index management and status."""

from __future__ import annotations

import logging
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.events import ColumnDef, IndexCleared, TableOutput
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.models import SnapshotTarget
from rbtr.workspace import resolve_path

if TYPE_CHECKING:
    from .core import Engine

log = logging.getLogger(__name__)


class _Sub(StrEnum):
    STATUS = "status"
    CLEAR = "clear"
    REBUILD = "rebuild"
    PRUNE = "prune"
    MODEL = "model"
    SEARCH = "search"
    SEARCH_DIAG = "search-diag"


def cmd_index(engine: Engine, args: str) -> None:
    """Dispatch /index subcommands."""
    parts = args.strip().split(maxsplit=1)
    sub = parts[0].lower() if parts else _Sub.STATUS
    rest = parts[1] if len(parts) > 1 else ""

    match sub:
        case _Sub.STATUS | "":
            _status(engine)
        case _Sub.CLEAR:
            _clear(engine)
        case _Sub.REBUILD:
            _rebuild(engine)
        case _Sub.PRUNE:
            _prune(engine)
        case _Sub.MODEL:
            _model(engine, rest)
        case _Sub.SEARCH:
            _search(engine, rest)
        case _Sub.SEARCH_DIAG:
            _search_diag(engine, rest)
        case _:
            engine._warn(f"Unknown subcommand: {sub}")
            engine._out(
                "Usage: /index [status | clear | rebuild | prune"
                " | model <id> | search <query> | search-diag <query>]",
            )


_MAX_COMMIT_ROWS = 20
"""Maximum commits to show in per-commit breakdown."""


def _status(engine: Engine) -> None:
    """Show index stats with snapshot comparison and per-commit breakdown."""
    store = engine.state.index
    if store is None:
        engine._out("No index loaded. Use /review to start indexing.")
        return

    target = engine.state.review_target
    if target is None:
        engine._out("No review target selected.")
        return

    head_ref = target.head_commit

    head_chunks = store.get_chunks(head_ref)
    head_edges = store.get_edges(head_ref)

    if isinstance(target, SnapshotTarget):
        base_chunks: list[Chunk] = []
        base_edges: list[Edge] = []
    else:
        base_ref = target.base_commit
        base_chunks = store.get_chunks(base_ref)
        base_edges = store.get_edges(base_ref)

    # DB file size.
    db_path = resolve_path(config.index.db_dir) / "index.duckdb"
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB"
    else:
        size_str = "in-memory"

    orphans = store.count_orphan_chunks()

    if isinstance(target, SnapshotTarget):
        engine._out(f"Target: {target.ref_label}")
    else:
        engine._out(f"Target: {target.base_branch} → {target.head_branch}")
    if not head_chunks and base_chunks:
        engine._out(f"Showing: {base_ref} (head not indexed yet)")
    engine._out(f"DB size: {size_str}")
    if orphans > 0:
        engine._out(f"Orphaned chunks: {orphans} (will be pruned on next index)")

    # ── Snapshot comparison (diff targets only) ──────────────────
    if not isinstance(target, SnapshotTarget):
        _emit_snapshot_table(engine, base_chunks, head_chunks, base_edges, head_edges)

    # ── Per-kind breakdown (head, or base if head not indexed) ──
    display_chunks = head_chunks if head_chunks else base_chunks
    display_edges = head_edges if head_chunks else base_edges
    if not display_chunks:
        engine._out("No chunks indexed yet.")
        return
    _emit_chunk_table(engine, display_chunks)
    _emit_edge_table(engine, display_edges)

    # ── Per-commit breakdown ─────────────────────────────────────
    if head_chunks:
        _emit_commit_table(engine, head_chunks, base_ref, head_ref)

    n_files = len({c.file_path for c in display_chunks})
    status = "ready" if engine.state.index_ready else "building"
    engine._context(
        f"[/index → {len(display_chunks)} symbols]",
        f"Index status: {n_files} files, {len(display_chunks)} symbols, {status}.",
    )


def _emit_snapshot_table(
    engine: Engine,
    base_chunks: list[Chunk],
    head_chunks: list[Chunk],
    base_edges: list[Edge],
    head_edges: list[Edge],
) -> None:
    """Emit a side-by-side comparison of base and head snapshots."""

    def _stats(chunks: list[Chunk], edges: list[Edge]) -> tuple[str, str, str]:
        if not chunks:
            return ("-", "-", "-")
        files = len({c.file_path for c in chunks})
        return (str(files), str(len(chunks)), str(len(edges)))

    b_files, b_chunks, b_edges = _stats(base_chunks, base_edges)
    h_files, h_chunks, h_edges = _stats(head_chunks, head_edges)

    engine._emit(
        TableOutput(
            title="Snapshots",
            columns=[
                ColumnDef(header="", width=10),
                ColumnDef(header="Files", width=10),
                ColumnDef(header="Symbols", width=10),
                ColumnDef(header="Edges", width=10),
            ],
            rows=[
                ["base", b_files, b_chunks, b_edges],
                ["head", h_files, h_chunks, h_edges],
            ],
        )
    )


def _emit_commit_table(
    engine: Engine,
    head_chunks: list[Chunk],
    base_ref: str,
    head_ref: str,
) -> None:
    """Emit per-commit file and symbol counts for the review range.

    For each commit between base and head, diffs against its parent
    to find changed files, then counts how many head-snapshot symbols
    live in those files.
    """
    import pygit2

    from rbtr.git.objects import commit_log_between, resolve_commit

    repo = engine.state.repo
    if repo is None:
        return

    try:
        entries = commit_log_between(repo, base_ref, head_ref)
    except KeyError:
        return

    if not entries:
        return

    # Build file → chunk count map from head snapshot.
    file_symbol_counts: Counter[str] = Counter()
    for c in head_chunks:
        file_symbol_counts[c.file_path] += 1

    rows: list[list[str]] = []
    for entry in entries[:_MAX_COMMIT_ROWS]:
        try:
            commit = resolve_commit(repo, entry.sha)
        except KeyError:
            continue
        if not commit.parent_ids:
            continue
        parent = repo.get(commit.parent_ids[0])
        if parent is None:
            continue
        parent_commit = parent.peel(pygit2.Commit)
        d = repo.diff(parent_commit, commit)

        files: set[str] = set()
        for patch in d:
            if patch is None:
                continue
            delta = patch.delta
            if delta.new_file.path:
                files.add(delta.new_file.path)
            if delta.old_file.path:
                files.add(delta.old_file.path)

        n_files = len(files)
        n_symbols = sum(file_symbol_counts.get(f, 0) for f in files)
        rows.append(
            [
                entry.sha[:8],
                entry.message[:50],
                str(n_files),
                str(n_symbols),
            ]
        )

    total = len(entries)
    title = f"Commits ({total})"
    if total > _MAX_COMMIT_ROWS:
        title += f" — showing first {_MAX_COMMIT_ROWS}"

    engine._emit(
        TableOutput(
            title=title,
            columns=[
                ColumnDef(header="SHA", width=10),
                ColumnDef(header="Message", width=50),
                ColumnDef(header="Files", width=7),
                ColumnDef(header="Symbols", width=9),
            ],
            rows=rows,
        )
    )


def _emit_chunk_table(engine: Engine, chunks: list[Chunk]) -> None:
    """Emit a table of chunk counts grouped by kind, with embedding coverage."""
    counts: Counter[ChunkKind] = Counter()
    embedded: Counter[ChunkKind] = Counter()
    for c in chunks:
        counts[c.kind] += 1
        if c.embedding:
            embedded[c.kind] += 1

    total_embedded = sum(embedded.values())
    rows = [[kind.value, str(count), str(embedded[kind])] for kind, count in counts.most_common()]
    rows.append(["total", str(len(chunks)), str(total_embedded)])
    engine._emit(
        TableOutput(
            title="Chunks",
            columns=[
                ColumnDef(header="Kind", width=20),
                ColumnDef(header="Count", width=10),
                ColumnDef(header="Embedded", width=10),
            ],
            rows=rows,
        )
    )


def _emit_edge_table(engine: Engine, edges: list[Edge]) -> None:
    """Emit a table of edge counts grouped by kind."""
    counts: Counter[EdgeKind] = Counter()
    for e in edges:
        counts[e.kind] += 1

    rows = [[kind.value, str(count)] for kind, count in counts.most_common()]
    rows.append(["total", str(len(edges))])
    engine._emit(
        TableOutput(
            title="Edges",
            columns=[
                ColumnDef(header="Kind", width=20),
                ColumnDef(header="Count", width=10),
            ],
            rows=rows,
        )
    )


def _clear(engine: Engine) -> None:
    """Delete the DuckDB file and reset state.index."""
    store = engine.state.index
    if store is not None:
        store.close()
        engine.state.index = None
    engine.state.index_ready = False

    db_path = resolve_path(config.index.db_dir) / "index.duckdb"
    if db_path.exists():
        db_path.unlink()
        engine._out("Index cleared.")
    else:
        engine._out("No index file to clear.")
    engine._emit(IndexCleared())
    engine._context("[/index clear]", "Cleared the code index.")


def _rebuild(engine: Engine) -> None:
    """Clear and re-trigger a full index for the current review target."""
    if engine.state.review_target is None:
        engine._warn("No review target. Use /review first.")
        return

    # Clear first.
    _clear(engine)

    # Re-run indexing.
    from .indexing import run_index

    engine._out("Rebuilding index…")
    run_index(engine)
    engine._context("[/index rebuild]", "Rebuilt the code index.")


def _model(engine: Engine, model_id: str) -> None:
    """Show or change the embedding model."""
    if not model_id:
        engine._out(f"Embedding model: {config.index.embedding_model}")
        engine._out("Usage: /index model <org/repo/file.gguf>")
        return

    old_model = config.index.embedding_model
    if model_id == old_model:
        engine._out(f"Already using {model_id}.")
        return

    # Persist the new model to config.
    config.update(index={"embedding_model": model_id})
    engine._out(f"Embedding model: {old_model} → {model_id}")

    # Release the cached model so the next embed call loads the new one.
    try:
        from rbtr.index.embeddings import reset_model  # deferred: heavy native lib

        reset_model()
    except ImportError:
        pass  # llama-cpp not installed — nothing to release

    # Clear existing embeddings — they're from the old model.
    store = engine.state.index
    if store is None:
        engine._out("No index loaded. Embeddings will use the new model on next /review.")
        return

    cleared = store.clear_embeddings()
    store.checkpoint()
    if cleared:
        engine._out(f"Cleared {cleared} embeddings from old model.")

    # Re-embed with the new model if we have a review target.
    if engine.state.review_target is not None:
        from .indexing import run_index

        engine._out("Re-embedding with new model…")
        store.close()
        engine.state.index = None
        run_index(engine)


def _search(engine: Engine, query: str) -> None:
    """Run a search query and display the ranked results."""
    if not query:
        engine._out("Usage: /index search <query>")
        return

    store = engine.state.index
    if store is None:
        engine._out("No index loaded. Use /review to start indexing.")
        return

    target = engine.state.review_target
    if target is None:
        engine._out("No review target selected.")
        return

    results = store.search(target.head_commit, query, top_k=20)
    if not results:
        engine._out(f"No results for '{query}'.")
        return

    rows: list[list[str]] = []
    for r in results:
        c = r.chunk
        scope = f"{c.scope}." if c.scope else ""
        rows.append(
            [
                f"{r.score:.3f}",
                c.kind.value,
                f"{scope}{c.name}",
                f"{c.file_path}:{c.line_start}",
            ]
        )

    engine._emit(
        TableOutput(
            title=f"Search: {query}",
            columns=[
                ColumnDef(header="Score", width=7),
                ColumnDef(header="Kind", width=14),
                ColumnDef(header="Name", width=35),
                ColumnDef(header="Location", width=40),
            ],
            rows=rows,
        )
    )


def _search_diag(engine: Engine, query: str) -> None:
    """Run a search query and display the full signal breakdown."""
    if not query:
        engine._out("Usage: /index search-diag <query>")
        return

    store = engine.state.index
    if store is None:
        engine._out("No index loaded. Use /review to start indexing.")
        return

    target = engine.state.review_target
    if target is None:
        engine._out("No review target selected.")
        return

    from rbtr.index.search import classify_query, weights_for_query

    kind = classify_query(query)
    alpha, beta, gamma = weights_for_query(query)
    engine._out(
        f"Query: {query!r}  class={kind.value}"
        f"  weights: a(sem)={alpha:.2f} b(lex)={beta:.2f} g(name)={gamma:.2f}"
    )

    results = store.search(target.head_commit, query, top_k=20)
    if not results:
        engine._out(f"No results for '{query}'.")
        return

    rows: list[list[str]] = []
    for r in results:
        c = r.chunk
        rows.append(
            [
                f"{r.score:.3f}",
                f"{r.lexical:.2f}",
                f"{r.semantic:.2f}",
                f"{r.name:.2f}",
                f"{r.kind_boost:.1f}",
                f"{r.file_penalty:.1f}",
                f"{r.importance:.2f}",
                f"{r.proximity:.1f}",
                f"{c.file_path}:{c.name}",
            ]
        )

    engine._emit(
        TableOutput(
            title=f"Diagnostics: {query}",
            columns=[
                ColumnDef(header="Score", width=7),
                ColumnDef(header="Lex", width=5),
                ColumnDef(header="Sem", width=5),
                ColumnDef(header="Name", width=5),
                ColumnDef(header="Kind", width=5),
                ColumnDef(header="File", width=5),
                ColumnDef(header="Imp", width=5),
                ColumnDef(header="Prox", width=5),
                ColumnDef(header="Chunk", width=45),
            ],
            rows=rows,
        )
    )


def _prune(engine: Engine) -> None:
    """Manually prune orphaned chunks and edges."""
    store = engine.state.index
    if store is None:
        engine._out("No index loaded. Use /review to start indexing.")
        return

    chunks_deleted, edges_deleted = store.prune_orphans()
    if chunks_deleted or edges_deleted:
        store.checkpoint()
        engine._out(f"Pruned {chunks_deleted} orphan chunks, {edges_deleted} orphan edges.")
    else:
        engine._out("No orphans to prune.")
