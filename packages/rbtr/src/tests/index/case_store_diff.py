"""Scenarios for ``IndexStore.diff_chunks``.

Cases take the shared ``math_func`` and ``http_func`` fixtures
from ``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk


@dataclass(frozen=True)
class DiffScenario:
    """Declarative diff-family test data."""

    chunks: list[Chunk] = field(default_factory=list)
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)
    base: str = "base"
    head: str = "head"
    expected_added: list[str] = field(default_factory=list)
    expected_removed: list[str] = field(default_factory=list)
    expected_modified: list[str] = field(default_factory=list)


def case_no_changes(math_func: Chunk) -> DiffScenario:
    return DiffScenario(
        chunks=[math_func],
        snapshots=[
            ("base", math_func.file_path, math_func.blob_sha),
            ("head", math_func.file_path, math_func.blob_sha),
        ],
    )


def case_file_added_at_head(
    math_func: Chunk, http_func: Chunk
) -> DiffScenario:
    return DiffScenario(
        chunks=[math_func, http_func],
        snapshots=[
            ("base", math_func.file_path, math_func.blob_sha),
            ("head", math_func.file_path, math_func.blob_sha),
            ("head", http_func.file_path, http_func.blob_sha),
        ],
        expected_added=[http_func.id],
    )


def case_file_removed_at_head(
    math_func: Chunk, http_func: Chunk
) -> DiffScenario:
    return DiffScenario(
        chunks=[math_func, http_func],
        snapshots=[
            ("base", math_func.file_path, math_func.blob_sha),
            ("base", http_func.file_path, http_func.blob_sha),
            ("head", math_func.file_path, math_func.blob_sha),
        ],
        expected_removed=[http_func.id],
    )


def case_file_modified_between_base_and_head(math_func: Chunk) -> DiffScenario:
    updated = math_func.model_copy(
        update={"id": "math_1_v2", "blob_sha": "blob_math_v2"}
    )
    return DiffScenario(
        chunks=[math_func, updated],
        snapshots=[
            ("base", math_func.file_path, math_func.blob_sha),
            ("head", math_func.file_path, "blob_math_v2"),
        ],
        expected_modified=[updated.id],
    )


def case_mixed_added_removed_modified(
    math_func: Chunk, http_func: Chunk
) -> DiffScenario:
    updated = math_func.model_copy(
        update={"id": "math_1_v2", "blob_sha": "blob_math_v2"}
    )
    return DiffScenario(
        chunks=[math_func, updated, http_func],
        snapshots=[
            ("base", math_func.file_path, math_func.blob_sha),
            ("base", http_func.file_path, http_func.blob_sha),
            ("head", math_func.file_path, "blob_math_v2"),
            ("head", "src/new.py", "blob_new"),
        ],
        expected_modified=[updated.id],
        expected_removed=[http_func.id],
        expected_added=[],
    )
