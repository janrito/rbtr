"""Scenarios for chunk storage and retrieval.

Covers ``insert_chunks``, ``get_chunks`` (with every filter),
``has_blob``, ``count_chunks``, blob reuse across commits, and
upsert.

Each case returns a ``ChunkScenario`` describing what the store
should contain and what every assertion against the seeded store
expects.  A shared fixture in ``test_chunks.py`` turns the
scenario into a real ``IndexStore`` and yields ``(store,
scenario)`` for each test to assert against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk, ChunkKind

from tests.index.cases_common import (
    ALL_CHUNKS,
    HTTP_FUNC,
    MATH_CLASS,
    MATH_FUNC,
    STRING_FUNC,
)


@dataclass(frozen=True)
class RepoData:
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

    # Ordered list of repo paths.  The first is ``repo_id=1`` after
    # registration, the second is ``repo_id=2``, and so on.  An
    # empty list means no repos registered at all.
    repo_paths: list[str] = field(default_factory=lambda: ["/default"])
    # Data to load per repo.  The outer list lines up with
    # ``repo_paths``; missing entries default to empty.
    per_repo: list[RepoData] = field(default_factory=list)

    # Expectations, each scoped to a specific (repo_id, commit).
    # ``repo_id`` is 1-based and matches ``repo_paths`` ordering.
    # A missing key means "no expectation recorded for that query".
    expected_chunk_ids: dict[tuple[int, str], list[str]] = field(
        default_factory=dict
    )
    expected_by_path: dict[tuple[int, str, str], list[str]] = field(
        default_factory=dict
    )
    expected_by_kind: dict[tuple[int, str, ChunkKind], list[str]] = field(
        default_factory=dict
    )
    expected_by_name: dict[tuple[int, str, str], list[str]] = field(
        default_factory=dict
    )
    expected_has_blob: dict[tuple[int, str], bool] = field(
        default_factory=dict
    )
    expected_counts: dict[tuple[int, str], int] = field(
        default_factory=dict
    )


# ── Empty ────────────────────────────────────────────────────────────


def case_empty_store() -> ChunkScenario:
    """No repos, no chunks."""
    return ChunkScenario(
        repo_paths=[],
        expected_chunk_ids={},
    )


def case_single_repo_no_data() -> ChunkScenario:
    """Repo registered but empty: every query yields nothing."""
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[RepoData()],
        expected_chunk_ids={(1, "head"): []},
        expected_has_blob={(1, "blob_math"): False},
        expected_counts={(1, "head"): 0},
    )


# ── Rich single-repo dataset ─────────────────────────────────────────


def case_full_dataset_on_one_commit() -> ChunkScenario:
    """All four shared chunks present on commit ``head``.

    Exercises filters (path, kind, name) and blob dedup for
    ``MATH_FUNC``/``MATH_CLASS`` which share ``blob_math``.
    """
    snapshots = [("head", c.file_path, c.blob_sha) for c in ALL_CHUNKS]
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(inserts=[list(ALL_CHUNKS)], snapshots=snapshots)
        ],
        expected_chunk_ids={
            (1, "head"): sorted(c.id for c in ALL_CHUNKS),
            (1, "does_not_exist"): [],
        },
        expected_by_path={
            (1, "head", "src/api/client.py"): [HTTP_FUNC.id],
            (1, "head", "src/math_utils.py"): sorted(
                [MATH_FUNC.id, MATH_CLASS.id]
            ),
            (1, "head", "no/such/path.py"): [],
        },
        expected_by_kind={
            (1, "head", ChunkKind.CLASS): [MATH_CLASS.id],
            (1, "head", ChunkKind.FUNCTION): sorted(
                [MATH_FUNC.id, HTTP_FUNC.id, STRING_FUNC.id]
            ),
        },
        expected_by_name={
            (1, "head", "normalize_whitespace"): [STRING_FUNC.id],
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


def case_blob_reused_across_commits() -> ChunkScenario:
    """One blob, two commits — inserted once, visible at both."""
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                inserts=[[MATH_FUNC]],
                snapshots=[
                    ("commit_a", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
                    ("commit_b", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
                ],
            )
        ],
        expected_chunk_ids={
            (1, "commit_a"): [MATH_FUNC.id],
            (1, "commit_b"): [MATH_FUNC.id],
        },
        expected_counts={
            (1, "commit_a"): 1,
            (1, "commit_b"): 1,
        },
    )


# ── Upsert (same id, new content) ────────────────────────────────────


def case_upsert_replaces_content() -> ChunkScenario:
    """Insert MATH_FUNC then a variant with the same id.

    Scenario carries both versions; the fixture inserts them in
    order so the second overwrites the first.  Assertion is on
    the final content.
    """
    updated = MATH_FUNC.model_copy(update={"content": "def calculate(): return 42"})
    return ChunkScenario(
        repo_paths=["/r"],
        per_repo=[
            RepoData(
                inserts=[[MATH_FUNC], [updated]],
                snapshots=[("head", MATH_FUNC.file_path, MATH_FUNC.blob_sha)],
            )
        ],
        expected_by_name={
            (1, "head", MATH_FUNC.name): [MATH_FUNC.id],
        },
        # The test checks content separately via a small helper on
        # the scenario; capture the expected marker here.
        expected_chunk_ids={(1, "head"): [MATH_FUNC.id]},
    )


# ── Repo isolation ───────────────────────────────────────────────────


def case_two_repos_same_blob_sha_isolated() -> ChunkScenario:
    """Identical blob sha in two repos stays isolated.

    Demonstrates the ``(repo_id, blob_sha)`` partitioning: repo_a
    has MATH_FUNC at 'head', repo_b has no chunks at all.
    """
    snapshots_a = [("head", MATH_FUNC.file_path, MATH_FUNC.blob_sha)]
    return ChunkScenario(
        repo_paths=["/repo_a", "/repo_b"],
        per_repo=[
            RepoData(inserts=[[MATH_FUNC]], snapshots=snapshots_a),
            RepoData(),
        ],
        expected_chunk_ids={
            (1, "head"): [MATH_FUNC.id],
            (2, "head"): [],
        },
        expected_has_blob={
            (1, MATH_FUNC.blob_sha): True,
            (2, MATH_FUNC.blob_sha): False,
        },
        expected_counts={
            (1, "head"): 1,
            (2, "head"): 0,
        },
    )
