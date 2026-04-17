"""Scenarios for ``IndexStore.diff_chunks``.

Each case describes two snapshots (base and head) and the
expected (added, removed, modified) chunk ids returned.  The
fixture in ``test_store_diff.py`` materialises the store and
reads the diff back.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk

from tests.index.cases_common import HTTP_FUNC, MATH_FUNC


@dataclass(frozen=True)
class DiffScenario:
    """Declarative diff-family test data."""

    chunks: list[Chunk] = field(default_factory=list)
    # [(commit_sha, file_path, blob_sha)]
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)

    base: str = "base"
    head: str = "head"
    expected_added: list[str] = field(default_factory=list)
    expected_removed: list[str] = field(default_factory=list)
    expected_modified: list[str] = field(default_factory=list)


# ── No change ────────────────────────────────────────────────────────


def case_no_changes() -> DiffScenario:
    """Base and head point to the same blob \u2192 empty diff."""
    return DiffScenario(
        chunks=[MATH_FUNC],
        snapshots=[
            ("base", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("head", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
        ],
    )


# ── Added / removed / modified ───────────────────────────────────────


def case_file_added_at_head() -> DiffScenario:
    """Head has a file base does not; chunk for new file is added."""
    return DiffScenario(
        chunks=[MATH_FUNC, HTTP_FUNC],
        snapshots=[
            ("base", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("head", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("head", HTTP_FUNC.file_path, HTTP_FUNC.blob_sha),
        ],
        expected_added=[HTTP_FUNC.id],
    )


def case_file_removed_at_head() -> DiffScenario:
    """Base has a file head does not; chunk is removed."""
    return DiffScenario(
        chunks=[MATH_FUNC, HTTP_FUNC],
        snapshots=[
            ("base", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("base", HTTP_FUNC.file_path, HTTP_FUNC.blob_sha),
            ("head", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
        ],
        expected_removed=[HTTP_FUNC.id],
    )


def case_file_modified_between_base_and_head() -> DiffScenario:
    """Same path, different blob; chunk at head appears in ``modified``."""
    updated = MATH_FUNC.model_copy(
        update={"id": "math_1_v2", "blob_sha": "blob_math_v2"}
    )
    return DiffScenario(
        chunks=[MATH_FUNC, updated],
        snapshots=[
            ("base", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("head", MATH_FUNC.file_path, "blob_math_v2"),
        ],
        expected_modified=[updated.id],
    )


def case_mixed_added_removed_modified() -> DiffScenario:
    """One file added, one removed, one modified \u2014 all at once."""
    updated = MATH_FUNC.model_copy(
        update={"id": "math_1_v2", "blob_sha": "blob_math_v2"}
    )
    return DiffScenario(
        chunks=[MATH_FUNC, updated, HTTP_FUNC],
        snapshots=[
            # base: math v1, http
            ("base", MATH_FUNC.file_path, MATH_FUNC.blob_sha),
            ("base", HTTP_FUNC.file_path, HTTP_FUNC.blob_sha),
            # head: math v2, no http, adds new file
            ("head", MATH_FUNC.file_path, "blob_math_v2"),
            ("head", "src/new.py", "blob_new"),
        ],
        expected_modified=[updated.id],
        expected_removed=[HTTP_FUNC.id],
        # No chunk row exists for src/new.py, so no added chunk surfaces.
        expected_added=[],
    )
