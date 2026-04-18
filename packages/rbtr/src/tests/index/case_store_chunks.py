"""Scenarios for chunk storage and retrieval.

Cases take named chunk fixtures (``math_func``, ``http_func``,
``string_func``, ``math_class``, ``all_store_chunks``) defined in
``conftest.py``.  No module-level constants or helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk, ChunkKind


@dataclass(frozen=True)
class RepoChunks:
    """Chunks and snapshots to load into one repo.

    ``inserts`` is a list of chunk batches — each batch becomes
    one ``insert_chunks`` call.  Most scenarios only need one
    batch; upsert scenarios need two so the second writes over
    the first (UPSERT does not replace within a single batch).
    """

    inserts: list[list[Chunk]] = field(default_factory=list)
    # (commit_sha, file_path, blob_sha)
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkScenario:
    """Declarative store state for a chunk-family test."""

    repo_paths: list[str] = field(default_factory=lambda: ["/default"])
    per_repo: list[RepoChunks] = field(default_factory=list)

    expected_chunk_ids: dict[tuple[int, str], list[str]] = field(default_factory=dict)
    expected_by_path: dict[tuple[int, str, str], list[str]] = field(default_factory=dict)
    expected_by_kind: dict[tuple[int, str, ChunkKind], list[str]] = field(default_factory=dict)
    expected_by_name: dict[tuple[int, str, str], list[str]] = field(default_factory=dict)
    expected_has_blob: dict[tuple[int, str], bool] = field(default_factory=dict)
    expected_counts: dict[tuple[int, str], int] = field(default_factory=dict)


# ── Empty ────────────────────────────────────────────────────────────


def case_empty_store() -> ChunkScenario:
    return ChunkScenario(
        repo_paths=[],
        expected_chunk_ids={},
    )


def case_single_repo_no_data() -> ChunkScenario:
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[RepoChunks()],
        expected_chunk_ids={(1, "head"): []},
        expected_has_blob={(1, "blob_math"): False},
        expected_counts={(1, "head"): 0},
    )


# ── Rich single-repo dataset ─────────────────────────────────────────


def case_full_dataset_on_one_commit(
    math_func: Chunk,
    http_func: Chunk,
    string_func: Chunk,
    math_class: Chunk,
    all_store_chunks: list[Chunk],
) -> ChunkScenario:
    """All four shared chunks on commit ``head``.

    Exercises filters (path, kind, name) and blob dedup for
    ``math_func`` / ``math_class`` which share ``blob_math``.
    """
    snapshots = [("head", c.file_path, c.blob_sha) for c in all_store_chunks]
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[RepoChunks(inserts=[list(all_store_chunks)], snapshots=snapshots)],
        expected_chunk_ids={
            (1, "head"): sorted(c.id for c in all_store_chunks),
            (1, "does_not_exist"): [],
        },
        expected_by_path={
            (1, "head", "src/api/client.py"): [http_func.id],
            (1, "head", "src/math_utils.py"): sorted([math_func.id, math_class.id]),
            (1, "head", "no/such/path.py"): [],
        },
        expected_by_kind={
            (1, "head", ChunkKind.CLASS): [math_class.id],
            (1, "head", ChunkKind.FUNCTION): sorted([math_func.id, http_func.id, string_func.id]),
        },
        expected_by_name={
            (1, "head", "normalize_whitespace"): [string_func.id],
            (1, "head", "does_not_exist"): [],
        },
        expected_has_blob={
            (1, "blob_math"): True,
            (1, "blob_http"): True,
            (1, "blob_string"): True,
            (1, "nonexistent"): False,
        },
        expected_counts={
            (1, "head"): 4,
            (1, "other"): 0,
        },
    )


# ── Blob reuse across commits ────────────────────────────────────────


def case_blob_reused_across_commits(math_func: Chunk) -> ChunkScenario:
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoChunks(
                inserts=[[math_func]],
                snapshots=[
                    ("commit_a", math_func.file_path, math_func.blob_sha),
                    ("commit_b", math_func.file_path, math_func.blob_sha),
                ],
            )
        ],
        expected_chunk_ids={
            (1, "commit_a"): [math_func.id],
            (1, "commit_b"): [math_func.id],
        },
        expected_counts={
            (1, "commit_a"): 1,
            (1, "commit_b"): 1,
        },
    )


# ── Upsert ───────────────────────────────────────────────────────────


def case_upsert_replaces_content(math_func: Chunk) -> ChunkScenario:
    """Two batches with the same id so the second overwrites the first."""
    updated = math_func.model_copy(update={"content": "def calculate(): return 42"})
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoChunks(
                inserts=[[math_func], [updated]],
                snapshots=[("head", math_func.file_path, math_func.blob_sha)],
            )
        ],
        expected_by_name={
            (1, "head", math_func.name): [math_func.id],
        },
        expected_chunk_ids={(1, "head"): [math_func.id]},
    )


# ── Repo isolation ───────────────────────────────────────────────────


def case_two_repos_same_blob_sha_isolated(math_func: Chunk) -> ChunkScenario:
    """repo_a has ``math_func``; repo_b has nothing.  has_blob scoped."""
    snapshots_a = [("head", math_func.file_path, math_func.blob_sha)]
    return ChunkScenario(
        repo_paths=["/repo_a", "/repo_b"],
        per_repo=[
            RepoChunks(inserts=[[math_func]], snapshots=snapshots_a),
            RepoChunks(),
        ],
        expected_chunk_ids={
            (1, "head"): [math_func.id],
            (2, "head"): [],
        },
        expected_has_blob={
            (1, math_func.blob_sha): True,
            (2, math_func.blob_sha): False,
        },
        expected_counts={
            (1, "head"): 1,
            (2, "head"): 0,
        },
    )
