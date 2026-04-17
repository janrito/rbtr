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

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind

from tests.index.cases_common import HTTP_FUNC, MATH_FUNC, STRING_FUNC


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
    # Ids that must remain after pruning.
    expected_surviving_chunk_ids: list[str] = field(default_factory=list)


# ── Clean store ──────────────────────────────────────────────────────


def case_all_chunks_referenced() -> PruneScenario:
    return PruneScenario(
        chunks=[MATH_FUNC, HTTP_FUNC],
        snapshots=[
            ("sha1", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("sha1", HTTP_FUNC.file_path, HTTP_FUNC.blob_sha),
        ],
        expected_orphan_count_before=0,
        expected_orphan_count_after=0,
        expected_surviving_chunk_ids=[MATH_FUNC.id, HTTP_FUNC.id],
    )


def case_noop_when_clean() -> PruneScenario:
    """Chunk + snapshot + edge all consistent \u2192 prune is a no-op."""
    return PruneScenario(
        chunks=[MATH_FUNC],
        snapshots=[("sha1", MATH_FUNC.file_path, MATH_FUNC.blob_sha)],
        edges=[
            (
                "sha1",
                Edge(source_id=MATH_FUNC.id, target_id=MATH_FUNC.id, kind=EdgeKind.IMPORTS),
            ),
        ],
        expected_orphan_count_before=0,
        expected_prune_chunks=0,
        expected_prune_edges=0,
        expected_surviving_chunk_ids=[MATH_FUNC.id],
    )


# ── Unreferenced chunks ──────────────────────────────────────────────


def case_chunks_without_snapshots_are_orphaned() -> PruneScenario:
    return PruneScenario(
        chunks=[MATH_FUNC, HTTP_FUNC, STRING_FUNC],
        snapshots=[("sha1", MATH_FUNC.file_path, MATH_FUNC.blob_sha)],
        expected_orphan_count_before=2,
        expected_prune_chunks=2,
        expected_orphan_count_after=0,
        expected_surviving_chunk_ids=[MATH_FUNC.id],
    )


# ── Unreferenced edges ───────────────────────────────────────────────


def case_edges_on_commit_without_snapshots_are_pruned() -> PruneScenario:
    return PruneScenario(
        chunks=[MATH_FUNC, HTTP_FUNC],
        snapshots=[
            ("sha1", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("sha1", HTTP_FUNC.file_path, HTTP_FUNC.blob_sha),
        ],
        edges=[
            (
                "sha1",
                Edge(source_id=MATH_FUNC.id, target_id=HTTP_FUNC.id, kind=EdgeKind.IMPORTS),
            ),
            (
                "stale_sha",
                Edge(source_id=MATH_FUNC.id, target_id=HTTP_FUNC.id, kind=EdgeKind.IMPORTS),
            ),
        ],
        expected_prune_chunks=0,
        expected_prune_edges=1,
        expected_surviving_chunk_ids=[MATH_FUNC.id, HTTP_FUNC.id],
    )


# ── Path rename: same blob, different path ──────────────────────────


def case_file_renamed_old_chunk_pruned() -> PruneScenario:
    """Old chunk's (blob, path) no longer referenced: pruned.

    prune_orphans uses ``(blob_sha, file_path)``, so a chunk whose
    blob is reused at a *different* path does not protect the old
    record.
    """
    renamed = MATH_FUNC.model_copy(
        update={"id": "math_renamed", "file_path": "src/renamed.py"}
    )
    return PruneScenario(
        chunks=[MATH_FUNC, renamed],
        # Only the renamed path is snapshotted.
        snapshots=[("sha1", renamed.file_path, MATH_FUNC.blob_sha)],
        expected_orphan_count_before=1,
        expected_prune_chunks=1,
        expected_surviving_chunk_ids=[renamed.id],
    )
