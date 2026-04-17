"""Scenarios for ``IndexStore`` search and embedding methods.

One rich scenario family covers ``search_by_name``,
``search_similar``, ``search_fulltext``, the batch embedding
methods, and the FTS lifecycle (auto-rebuild).

Each case returns a ``SearchScenario`` describing what the
store contains and what every public read method should return.
A shared fixture in ``test_store_search.py`` seeds the store
and yields ``(store, scenario)``.
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
    VEC_HTTP,
    VEC_MATH,
    VEC_STRING,
)


@dataclass(frozen=True)
class SearchScenario:
    """Declarative search-family test data."""

    chunks: list[Chunk] = field(default_factory=list)
    # [(commit_sha, file_path, blob_sha)]
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)
    # {chunk_id: vector}
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    commit: str = "head"

    # search_by_name expectations: (pattern, expected ids in result order).
    by_name: list[tuple[str, list[str]]] = field(default_factory=list)

    # search_similar expectations:
    #   (query_vector, top_k, id_order_constraint).
    # id_order_constraint lists ids whose ordering the test must
    # respect.  Ids not mentioned may appear anywhere.  An empty
    # constraint with top_k=None simply asserts result count.
    similar: list[tuple[list[float], int, list[str]]] = field(default_factory=list)

    # Queries whose top-ranked result id must match.
    fulltext_top: list[tuple[str, str]] = field(default_factory=list)

    # Queries expected to yield zero hits.
    fulltext_empty: list[tuple[str, str]] = field(default_factory=list)


# ── Rich shared scenario ─────────────────────────────────────────────


def case_full_dataset_with_embeddings() -> SearchScenario:
    """Four chunks on ``head``, each embedded on its own axis."""
    return SearchScenario(
        chunks=list(ALL_CHUNKS),
        snapshots=[("head", c.file_path, c.blob_sha) for c in ALL_CHUNKS],
        embeddings={
            MATH_FUNC.id: VEC_MATH,
            HTTP_FUNC.id: VEC_HTTP,
            STRING_FUNC.id: VEC_STRING,
            MATH_CLASS.id: VEC_MATH,
        },
        by_name=[
            ("standard_deviation", [MATH_FUNC.id]),
            ("NORMALIZE", [STRING_FUNC.id]),       # case-insensitive
            ("fetch_json", [HTTP_FUNC.id]),
            ("zzz_nonexistent", []),
        ],
        similar=[
            # Query near math axis \u2192 math chunks first.
            ([0.9, 0.1, 0.0], 4, [MATH_FUNC.id, MATH_CLASS.id, STRING_FUNC.id]),
            # Exact HTTP match \u2192 HTTP first.
            (VEC_HTTP, 1, [HTTP_FUNC.id]),
            # Nonexistent commit \u2192 empty regardless of vector.
        ],
        fulltext_top=[
            ("variance", MATH_FUNC.id),
            ("endpoint", HTTP_FUNC.id),
            ("whitespace", STRING_FUNC.id),
        ],
        fulltext_empty=[
            ("zzz_gibberish_xyz", "head"),
        ],
    )


def case_full_dataset_without_embeddings() -> SearchScenario:
    """Same chunks, no embeddings \u2014 similarity hits nothing."""
    return SearchScenario(
        chunks=list(ALL_CHUNKS),
        snapshots=[("head", c.file_path, c.blob_sha) for c in ALL_CHUNKS],
        by_name=[("fetch_json", [HTTP_FUNC.id])],
        similar=[(VEC_MATH, 10, [])],  # empty list \u2194 top_k=10 yields no rows
        fulltext_top=[("variance", MATH_FUNC.id)],
    )


def case_partial_embeddings() -> SearchScenario:
    """Only one chunk embedded; similarity returns it alone."""
    return SearchScenario(
        chunks=list(ALL_CHUNKS),
        snapshots=[("head", c.file_path, c.blob_sha) for c in ALL_CHUNKS],
        embeddings={MATH_FUNC.id: VEC_MATH},
        similar=[(VEC_MATH, 10, [MATH_FUNC.id])],
    )


def case_alternative_commit_has_nothing() -> SearchScenario:
    """Chunks on 'head'; 'other' returns empty for every query."""
    return SearchScenario(
        chunks=list(ALL_CHUNKS),
        snapshots=[("head", c.file_path, c.blob_sha) for c in ALL_CHUNKS],
        embeddings={MATH_FUNC.id: VEC_MATH},
        commit="other",
        by_name=[("standard_deviation", [])],
        similar=[(VEC_MATH, 4, [])],
        fulltext_empty=[("variance", "other")],
    )
