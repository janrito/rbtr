"""Background indexing — triggered by /review, emits progress events."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.events import IndexProgress, IndexReady, IndexStarted, Output, OutputLevel
from rbtr.git import FileEntry, list_files
from rbtr.index.orchestrator import build_index, update_index
from rbtr.index.store import IndexStore
from rbtr.models import SnapshotTarget
from rbtr.plugins.manager import get_manager

if TYPE_CHECKING:
    from .core import Engine

log = logging.getLogger(__name__)


def run_index(engine: Engine) -> None:
    """Start background indexing for the current review target.

    Spawns a daemon thread so the calling command (`/review`)
    returns immediately.  The user can keep chatting while the
    index builds; tools that need the index are hidden until
    `IndexReady` is emitted.
    """
    t = threading.Thread(target=_build_index, args=(engine,), daemon=True)
    t.start()


def _build_index(engine: Engine) -> None:
    """Build or update the code index (runs in a background thread)."""
    target = engine.state.review_target
    repo = engine.state.repo
    if target is None or repo is None:
        return

    if not config.index.enabled:
        return

    head_ref = target.head_commit

    # Open (or reuse) the DuckDB store.
    db_path = Path(config.index.db_dir) / "index.duckdb"
    store = IndexStore(db_path)
    engine.state.index = store
    engine.state.index_ready = False

    # Count files for the progress total.
    head_files = list(
        list_files(
            repo,
            head_ref,
            max_file_size=config.index.max_file_size,
            include=config.index.include,
            exclude=config.index.extend_exclude,
        )
    )
    total = len(head_files)
    engine._emit(IndexStarted(total_files=total))

    # Warn about missing grammars (6.2).
    _warn_missing_grammars(engine, head_files)

    # Progress callbacks wired to events — one per phase.
    def on_parse_progress(done: int, total: int) -> None:
        engine._emit(IndexProgress(phase="parsing", indexed=done, total=total))

    def on_embed_progress(done: int, total: int) -> None:
        engine._emit(IndexProgress(phase="embedding", indexed=done, total=total))

    # Snapshots index a single commit; diff targets index base
    # first and then incrementally update for head.
    try:
        if isinstance(target, SnapshotTarget):
            result = build_index(
                repo,
                head_ref,
                store,
                on_progress=on_parse_progress,
                on_embed_progress=on_embed_progress,
            )
        else:
            base_ref = target.base_commit
            build_index(
                repo,
                base_ref,
                store,
                on_progress=on_parse_progress,
                on_embed_progress=on_embed_progress,
            )

            result = update_index(
                repo,
                base_ref,
                head_ref,
                store,
                on_progress=on_parse_progress,
                on_embed_progress=on_embed_progress,
            )

        engine.state.index_ready = True
        engine._emit(IndexReady(chunk_count=result.stats.total_chunks))
        engine._emit(
            Output(
                text=(
                    f"Index ready: {result.stats.total_chunks} symbols, "
                    f"{result.stats.total_edges} edges, "
                    f"{result.stats.elapsed_seconds:.1f}s"
                ),
            )
        )
        engine._context(
            f"[index ready → {result.stats.total_chunks} symbols]",
            f"Code index built: {result.stats.total_chunks} symbols, "
            f"{result.stats.total_edges} edges.",
        )

        if result.errors:
            for err in result.errors:
                engine._emit(Output(text=err, level=OutputLevel.WARNING))

    except Exception as exc:
        log.exception("Indexing failed")
        engine._emit(
            Output(
                text=f"Indexing failed: {exc!r}\nReview continues without index.",
                level=OutputLevel.WARNING,
            )
        )
        engine.state.index = None
        store.close()


def _warn_missing_grammars(engine: Engine, files: list[FileEntry]) -> None:
    """Emit a warning listing languages detected but without a grammar."""
    mgr = get_manager()
    missing: set[str] = set()

    for f in files:
        lang_id = mgr.detect_language(f.path)
        if lang_id is not None and mgr.missing_grammar(lang_id):
            missing.add(lang_id)

    if missing:
        names = ", ".join(sorted(missing))
        engine._emit(
            Output(
                text=f"Missing grammars (falling back to plaintext): {names}",
                level=OutputLevel.WARNING,
            )
        )
