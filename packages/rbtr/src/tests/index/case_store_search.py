"""Scenarios for ``IndexStore`` search and embedding methods.

Cases take named chunk / embedding vector fixtures from
``conftest.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import Chunk


@dataclass(frozen=True)
class SearchScenario:
    """Declarative search-family test data."""

    chunks: list[Chunk] = field(default_factory=list)
    snapshots: list[tuple[str, str, str]] = field(default_factory=list)
    embeddings: dict[str, list[float]] = field(default_factory=dict)

    commit: str = "head"

    by_name: list[tuple[str, list[str]]] = field(default_factory=list)
    similar: list[tuple[list[float], int, list[str]]] = field(default_factory=list)
    fulltext_top: list[tuple[str, str]] = field(default_factory=list)
    fulltext_empty: list[tuple[str, str]] = field(default_factory=list)


def case_full_dataset_with_embeddings(
    math_func: Chunk,
    http_func: Chunk,
    string_func: Chunk,
    math_class: Chunk,
    all_store_chunks: list[Chunk],
    vec_math: list[float],
    vec_http: list[float],
    vec_string: list[float],
) -> SearchScenario:
    """Four chunks on ``head``, each embedded on its own axis."""
    return SearchScenario(
        chunks=list(all_store_chunks),
        snapshots=[("head", c.file_path, c.blob_sha) for c in all_store_chunks],
        embeddings={
            math_func.id: vec_math,
            http_func.id: vec_http,
            string_func.id: vec_string,
            math_class.id: vec_math,
        },
        by_name=[
            ("standard_deviation", [math_func.id]),
            ("NORMALIZE", [string_func.id]),
            ("fetch_json", [http_func.id]),
            ("zzz_nonexistent", []),
        ],
        similar=[
            ([0.9, 0.1, 0.0], 4, [math_func.id, math_class.id, string_func.id]),
            (vec_http, 1, [http_func.id]),
        ],
        fulltext_top=[
            ("variance", math_func.id),
            ("endpoint", http_func.id),
            ("whitespace", string_func.id),
        ],
        fulltext_empty=[
            ("zzz_gibberish_xyz", "head"),
        ],
    )


def case_full_dataset_without_embeddings(
    math_func: Chunk,
    http_func: Chunk,
    all_store_chunks: list[Chunk],
    vec_math: list[float],
) -> SearchScenario:
    """Same chunks, no embeddings — similarity hits nothing."""
    return SearchScenario(
        chunks=list(all_store_chunks),
        snapshots=[("head", c.file_path, c.blob_sha) for c in all_store_chunks],
        by_name=[("fetch_json", [http_func.id])],
        similar=[(vec_math, 10, [])],
        fulltext_top=[("variance", math_func.id)],
    )


def case_partial_embeddings(
    math_func: Chunk,
    all_store_chunks: list[Chunk],
    vec_math: list[float],
) -> SearchScenario:
    """Only one chunk embedded; similarity returns it alone."""
    return SearchScenario(
        chunks=list(all_store_chunks),
        snapshots=[("head", c.file_path, c.blob_sha) for c in all_store_chunks],
        embeddings={math_func.id: vec_math},
        similar=[(vec_math, 10, [math_func.id])],
    )


def case_alternative_commit_has_nothing(
    math_func: Chunk,
    all_store_chunks: list[Chunk],
    vec_math: list[float],
) -> SearchScenario:
    """Chunks on 'head'; 'other' returns empty for every query."""
    return SearchScenario(
        chunks=list(all_store_chunks),
        snapshots=[("head", c.file_path, c.blob_sha) for c in all_store_chunks],
        embeddings={math_func.id: vec_math},
        commit="other",
        by_name=[("standard_deviation", [])],
        similar=[(vec_math, 4, [])],
        fulltext_empty=[("variance", "other")],
    )
