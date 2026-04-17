"""Scenarios for ``IndexStore.prune_orphans`` and ``count_orphan_chunks``.

``prune_orphans`` uses a tighter invariant than
``sweep_orphan_chunks``: a chunk is kept iff
``(blob_sha, file_path)`` is referenced by some snapshot.  So a
chunk whose blob still exists but whose original path no longer
appears in any snapshot is still pruned.  These cases capture
that semantic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk, Edge, EdgeKind


@dataclass(frozen=True)
class PruneScenario:
    """Declarative prune-family test data."""

    chunks: list[Chunk] = field(default_factory=list)
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)
    edges: list[tuple[str, Edge]] = field(default_factory=list)

    expected_orphan_count_before: int = 0
    expected_prune_chunks: int = 0
    expected_prune_edges: int = 0
    expected_orphan_count_after: int = 0
    expected_surviving_chunk_ids: list[str] = field(default_factory=list)


def case_all_chunks_referenced(
    math_func: Chunk, http_func: Chunk
) -> PruneScenario:
    return PruneScenario(
        chunks=[math_func, http_func],
        snapshots=[
            ("sha1", math_func.file_path, math_func.blob_sha),
            ("sha1", http_func.file_path, http_func.blob_sha),
        ],
        expected_surviving_chunk_ids=[math_func.id, http_func.id],
    )


def case_noop_when_clean(math_func: Chunk) -> PruneScenario:
    """Chunk + snapshot + edge all consistent → prune is a no-op."""
    return PruneScenario(
        chunks=[math_func],
        snapshots=[("sha1", math_func.file_path, math_func.blob_sha)],
        edges=[
            (
                "sha1",
                Edge(source_id=math_func.id, target_id=math_func.id, kind=EdgeKind.IMPORTS),
            ),
        ],
        expected_surviving_chunk_ids=[math_func.id],
    )


def case_chunks_without_snapshots_are_orphaned(
    math_func: Chunk, http_func: Chunk, string_func: Chunk
) -> PruneScenario:
    return PruneScenario(
        chunks=[math_func, http_func, string_func],
        snapshots=[("sha1", math_func.file_path, math_func.blob_sha)],
        expected_orphan_count_before=2,
        expected_prune_chunks=2,
        expected_surviving_chunk_ids=[math_func.id],
    )


def case_edges_on_commit_without_snapshots_are_pruned(
    math_func: Chunk, http_func: Chunk
) -> PruneScenario:
    return PruneScenario(
        chunks=[math_func, http_func],
        snapshots=[
            ("sha1", math_func.file_path, math_func.blob_sha),
            ("sha1", http_func.file_path, http_func.blob_sha),
        ],
        edges=[
            (
                "sha1",
                Edge(source_id=math_func.id, target_id=http_func.id, kind=EdgeKind.IMPORTS),
            ),
            (
                "stale_sha",
                Edge(source_id=math_func.id, target_id=http_func.id, kind=EdgeKind.IMPORTS),
            ),
        ],
        expected_prune_edges=1,
        expected_surviving_chunk_ids=[math_func.id, http_func.id],
    )


def case_file_renamed_old_chunk_pruned(math_func: Chunk) -> PruneScenario:
    """Blob reused at a different path; old chunk record is pruned."""
    renamed = math_func.model_copy(
        update={"id": "math_renamed", "file_path": "src/renamed.py"}
    )
    return PruneScenario(
        chunks=[math_func, renamed],
        snapshots=[("sha1", renamed.file_path, math_func.blob_sha)],
        expected_orphan_count_before=1,
        expected_prune_chunks=1,
        expected_surviving_chunk_ids=[renamed.id],
    )
