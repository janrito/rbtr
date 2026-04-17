"""Scenarios for ``IndexStore`` GC primitives.

Covers ``drop_commit``, ``sweep_orphan_chunks``,
``sweep_orphan_commits``, and ``prune_orphans``.  Cases take the
three blob-distinct ``gc_chunk_{x,y,z}`` fixtures from
``conftest.py``; no module-level constants or helpers live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk, Edge, EdgeKind


@dataclass(frozen=True)
class RepoData:
    chunks: list[Chunk] = field(default_factory=list)
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)
    edges: list[tuple[str, Edge]] = field(default_factory=list)
    marked: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GcScenario:
    repo_paths: list[str] = field(default_factory=lambda: ["/default"])
    per_repo: list[RepoData] = field(default_factory=list)

    drop: list[
        tuple[int, str, dict[str, int | list[str]]]
    ] = field(default_factory=list)
    sweep_chunks: list[tuple[int, int]] = field(default_factory=list)
    sweep_commits: list[
        tuple[int, dict[str, int], list[str]]
    ] = field(default_factory=list)


def case_two_commits_sharing_one_blob(
    gc_chunk_x: Chunk, gc_chunk_y: Chunk, gc_chunk_z: Chunk
) -> GcScenario:
    """commit_a: blobs x, y.  commit_b: blobs y, z.  Both marked."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[gc_chunk_x, gc_chunk_y, gc_chunk_z],
                snapshots=[
                    ("commit_a", "x.py", "blob_x"),
                    ("commit_a", "y.py", "blob_y"),
                    ("commit_b", "y.py", "blob_y"),
                    ("commit_b", "z.py", "blob_z"),
                ],
                edges=[
                    (
                        "commit_a",
                        Edge(source_id="cx", target_id="cy", kind=EdgeKind.IMPORTS),
                    ),
                    (
                        "commit_b",
                        Edge(source_id="cy", target_id="cz", kind=EdgeKind.IMPORTS),
                    ),
                ],
                marked=["commit_a", "commit_b"],
            )
        ],
        drop=[
            (
                1,
                "commit_a",
                {
                    "commits": 1,
                    "snapshots": 2,
                    "edges": 1,
                    "chunks": 1,
                    "gone_commit_ids": ["commit_a"],
                    "surviving_chunk_ids": ["cy", "cz"],
                },
            )
        ],
        sweep_chunks=[(1, 0)],
        sweep_commits=[(1, {"snapshots": 0, "edges": 0, "chunks": 0}, ["commit_a", "commit_b"])],
    )


def case_drop_same_commit_twice(gc_chunk_x: Chunk) -> GcScenario:
    """Dropping a commit twice — second call reports zeroes."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[gc_chunk_x],
                snapshots=[("commit_a", "x.py", "blob_x")],
                marked=["commit_a"],
            )
        ],
        drop=[
            (1, "commit_a", {"commits": 1}),
            (1, "commit_a", {"commits": 0, "snapshots": 0}),
        ],
    )


def case_drop_scoped_to_target_repo() -> GcScenario:
    """Same sha marked in two repos; dropping in one leaves the other."""
    return GcScenario(
        repo_paths=["/r1", "/r2"],
        per_repo=[
            RepoData(marked=["shared_sha"]),
            RepoData(marked=["shared_sha"]),
        ],
        drop=[
            (
                1,
                "shared_sha",
                {
                    "gone_commit_ids": ["shared_sha"],
                    "surviving_chunk_ids": [],
                },
            )
        ],
    )


def case_orphan_chunk_without_snapshot(
    gc_chunk_x: Chunk, gc_chunk_y: Chunk
) -> GcScenario:
    """Chunk inserted but never snapshotted → sweep_orphan_chunks = 1."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[gc_chunk_x, gc_chunk_y],
                snapshots=[("head", "y.py", "blob_y")],
            )
        ],
        sweep_chunks=[(1, 1)],
    )


def case_crashed_build_leaves_residue(gc_chunk_x: Chunk) -> GcScenario:
    """Snapshot + edge written but mark_indexed never called."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[gc_chunk_x],
                snapshots=[("crashed_sha", "x.py", "blob_x")],
                edges=[
                    (
                        "crashed_sha",
                        Edge(source_id="cx", target_id="other", kind=EdgeKind.IMPORTS),
                    ),
                ],
                marked=[],
            )
        ],
        sweep_commits=[
            (1, {"snapshots": 1, "edges": 1, "chunks": 1}, []),
        ],
    )


def case_mixed_completed_and_crashed(
    gc_chunk_x: Chunk, gc_chunk_y: Chunk
) -> GcScenario:
    """One good commit + one crashed residue: sweep only removes the crashed one."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[gc_chunk_x, gc_chunk_y],
                snapshots=[
                    ("good_sha", "x.py", "blob_x"),
                    ("crashed_sha", "y.py", "blob_y"),
                ],
                marked=["good_sha"],
            )
        ],
        sweep_commits=[
            (1, {"snapshots": 1, "chunks": 1}, ["good_sha"]),
        ],
    )
