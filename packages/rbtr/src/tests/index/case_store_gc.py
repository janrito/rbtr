"""Scenarios for ``IndexStore`` GC primitives.

Covers ``drop_commit``, ``sweep_orphan_chunks``,
``sweep_orphan_commits``, and ``prune_orphans``.  Cases describe
a seeded store and the per-operation expectations.  Tests call
one operation per test function and assert against the relevant
expectation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind


# ── Building blocks ──────────────────────────────────────────────────
#
# Small chunks specific to GC; blob sharing is the point here so we
# do not reuse the shared MATH/HTTP/STRING constants.

_CHUNK_X = Chunk(
    id="cx",
    blob_sha="blob_x",
    file_path="x.py",
    kind=ChunkKind.FUNCTION,
    name="f_x",
    content="def f_x(): pass",
    line_start=1,
    line_end=1,
)
_CHUNK_Y = Chunk(
    id="cy",
    blob_sha="blob_y",
    file_path="y.py",
    kind=ChunkKind.FUNCTION,
    name="f_y",
    content="def f_y(): pass",
    line_start=1,
    line_end=1,
)
_CHUNK_Z = Chunk(
    id="cz",
    blob_sha="blob_z",
    file_path="z.py",
    kind=ChunkKind.FUNCTION,
    name="f_z",
    content="def f_z(): pass",
    line_start=1,
    line_end=1,
)


@dataclass(frozen=True)
class RepoData:
    chunks: list[Chunk] = field(default_factory=list)
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)  # (commit, path, blob)
    edges: list[tuple[str, Edge]] = field(default_factory=list)         # (commit, edge)
    marked: list[str] = field(default_factory=list)                      # indexed commit shas


@dataclass(frozen=True)
class GcScenario:
    repo_paths: list[str] = field(default_factory=lambda: ["/default"])
    per_repo: list[RepoData] = field(default_factory=list)

    # drop_commit expectations.
    drop: list[
        tuple[int, str, dict[str, int | list[str]]]
    ] = field(default_factory=list)
    # Each tuple is (repo_id, commit_to_drop, expectations):
    #   commits / snapshots / edges / chunks   \u2014 accurate counts
    #   gone_commit_ids     \u2014 list of commit_sha that should become unindexed
    #   surviving_chunk_ids \u2014 ids that must remain in another commit

    # sweep_orphan_chunks expectations: (repo_id, expected_deleted_count)
    sweep_chunks: list[tuple[int, int]] = field(default_factory=list)

    # sweep_orphan_commits expectations: (repo_id, {counts...}, surviving_commits)
    sweep_commits: list[
        tuple[int, dict[str, int], list[str]]
    ] = field(default_factory=list)


# ── Shared blob-aware scenario ───────────────────────────────────────


def case_two_commits_sharing_one_blob() -> GcScenario:
    """commit_a: blobs x, y.  commit_b: blobs y, z.  Both marked.

    Dropping commit_a preserves y (shared with commit_b) and
    deletes x's chunk (orphan).  Covers every drop_commit claim:
    ``has_indexed`` flip, own snapshots/edges gone, shared chunks
    survive, orphan chunks vanish, counts are accurate, per-repo
    scoping (another repo's same sha is untouched).
    """
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[_CHUNK_X, _CHUNK_Y, _CHUNK_Z],
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
        sweep_chunks=[(1, 0)],  # no orphans after healthy seed
        sweep_commits=[(1, {"snapshots": 0, "edges": 0, "chunks": 0}, ["commit_a", "commit_b"])],
    )


# ── Drop idempotence ────────────────────────────────────────────────


def case_drop_same_commit_twice() -> GcScenario:
    """Dropping a commit twice \u2014 second call reports zeroes."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[_CHUNK_X],
                snapshots=[("commit_a", "x.py", "blob_x")],
                marked=["commit_a"],
            )
        ],
        drop=[
            (1, "commit_a", {"commits": 1}),
            (1, "commit_a", {"commits": 0, "snapshots": 0}),
        ],
    )


# ── Per-repo scoping ────────────────────────────────────────────────


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
                    "surviving_chunk_ids": [],  # none to survive
                },
            )
        ],
    )


# ── Orphan chunks: inserted with no snapshot ────────────────────────


def case_orphan_chunk_without_snapshot() -> GcScenario:
    """Chunk inserted but never snapshotted \u2192 sweep_orphan_chunks = 1."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[_CHUNK_X, _CHUNK_Y],
                # Only y gets a snapshot.
                snapshots=[("head", "y.py", "blob_y")],
            )
        ],
        sweep_chunks=[(1, 1)],
    )


# \u2500\u2500 sweep_orphan_commits: crashed-build residue \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def case_crashed_build_leaves_residue() -> GcScenario:
    """Snapshot + edge written but mark_indexed never called.

    sweep_orphan_commits deletes the snapshot, the edge, and the
    now-orphan chunk in one transaction.
    """
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[_CHUNK_X],
                snapshots=[("crashed_sha", "x.py", "blob_x")],
                edges=[
                    (
                        "crashed_sha",
                        Edge(source_id="cx", target_id="other", kind=EdgeKind.IMPORTS),
                    ),
                ],
                marked=[],  # never completed
            )
        ],
        sweep_commits=[
            (
                1,
                {"snapshots": 1, "edges": 1, "chunks": 1},
                [],  # no surviving indexed commits
            )
        ],
    )


def case_mixed_completed_and_crashed() -> GcScenario:
    """One good commit + one crashed residue: sweep only removes the crashed one."""
    return GcScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                chunks=[_CHUNK_X, _CHUNK_Y],
                snapshots=[
                    ("good_sha", "x.py", "blob_x"),
                    ("crashed_sha", "y.py", "blob_y"),
                ],
                marked=["good_sha"],
            )
        ],
        sweep_commits=[
            (
                1,
                {"snapshots": 1, "chunks": 1},
                ["good_sha"],
            )
        ],
    )

