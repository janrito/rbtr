"""Handler for /index — index management and status."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.events import ColumnDef, IndexCleared, TableOutput
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.styles import STYLE_DIM

if TYPE_CHECKING:
    from .core import Engine


class _Sub(StrEnum):
    STATUS = "status"
    CLEAR = "clear"
    REBUILD = "rebuild"
    PRUNE = "prune"
    MODEL = "model"


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
        case _:
            engine._warn(f"Unknown subcommand: {sub}")
            engine._out(
                "Usage: /index [status | clear | rebuild | prune | model <id>]",
                style=STYLE_DIM,
            )


def _status(engine: Engine) -> None:
    """Show index stats with per-kind breakdowns."""
    store = engine.session.index
    if store is None:
        engine._out("No index loaded. Use /review to start indexing.")
        return

    target = engine.session.review_target
    if target is None:
        engine._out("No review target selected.")
        return

    base_ref = target.base_branch
    head_ref = target.head_branch

    # Query head first; fall back to base if head isn't indexed yet
    # (build_index finishes the base before update_index starts the head).
    chunks = store.get_chunks(head_ref)
    edges = store.get_edges(head_ref)
    showing_ref = head_ref

    if not chunks:
        base_chunks = store.get_chunks(base_ref)
        base_edges = store.get_edges(base_ref)
        if base_chunks:
            chunks = base_chunks
            edges = base_edges
            showing_ref = base_ref

    # DB file size.
    db_path = Path(config.index.db_dir) / "index.duckdb"
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB"
    else:
        size_str = "in-memory"

    orphans = store.count_orphan_chunks()

    engine._out(f"Target: {target.base_branch} → {target.head_branch}")
    if showing_ref != head_ref:
        engine._out(f"Showing: {showing_ref} (head not indexed yet)", style=STYLE_DIM)
    engine._out(f"DB size: {size_str}")
    if orphans > 0:
        engine._out(f"Orphaned chunks: {orphans} (will be pruned on next index)")

    # Chunk breakdown by kind.
    _emit_chunk_table(engine, chunks)

    # Edge breakdown by kind.
    _emit_edge_table(engine, edges)


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
    """Delete the DuckDB file and reset session.index."""
    store = engine.session.index
    if store is not None:
        store.close()
        engine.session.index = None
    engine.session.index_ready = False

    db_path = Path(config.index.db_dir) / "index.duckdb"
    if db_path.exists():
        db_path.unlink()
        engine._out("Index cleared.")
    else:
        engine._out("No index file to clear.")
    engine._emit(IndexCleared())


def _rebuild(engine: Engine) -> None:
    """Clear and re-trigger a full index for the current review target."""
    if engine.session.review_target is None:
        engine._warn("No review target. Use /review first.")
        return

    # Clear first.
    _clear(engine)

    # Re-run indexing.
    from .indexing import run_index

    engine._out("Rebuilding index…")
    run_index(engine)


def _model(engine: Engine, model_id: str) -> None:
    """Show or change the embedding model."""
    if not model_id:
        engine._out(f"Embedding model: {config.index.embedding_model}")
        engine._out("Usage: /index model <org/repo/file.gguf>", style=STYLE_DIM)
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
    store = engine.session.index
    if store is None:
        engine._out("No index loaded. Embeddings will use the new model on next /review.")
        return

    cleared = store.clear_embeddings()
    store.checkpoint()
    if cleared:
        engine._out(f"Cleared {cleared} embeddings from old model.")

    # Re-embed with the new model if we have a review target.
    if engine.session.review_target is not None:
        from .indexing import run_index

        engine._out("Re-embedding with new model…")
        store.close()
        engine.session.index = None
        run_index(engine)


def _prune(engine: Engine) -> None:
    """Manually prune orphaned chunks and edges."""
    store = engine.session.index
    if store is None:
        engine._out("No index loaded. Use /review to start indexing.")
        return

    chunks_deleted, edges_deleted = store.prune_orphans()
    if chunks_deleted or edges_deleted:
        store.checkpoint()
        engine._out(f"Pruned {chunks_deleted} orphan chunks, {edges_deleted} orphan edges.")
    else:
        engine._out("No orphans to prune.")
