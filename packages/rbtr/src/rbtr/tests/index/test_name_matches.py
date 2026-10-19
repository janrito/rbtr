"""Tiered name resolution for name_matches.

name_matches resolves a user-supplied symbol name to chunks.
It should prefer exact matches, falling back through tiers:
exact → case-insensitive exact → prefix → substring.
Only the best tier that has matches should be returned.
"""

from __future__ import annotations

import pytest

from rbtr.domain.models import ChunkKind, SnapshotRef
from rbtr.index.index import Index

from .conftest import make_chunk, seed_index

COMMIT = "head"


@pytest.fixture
def name_index(index: Index, head_ref: SnapshotRef) -> Index:
    """Store with chunks that create name ambiguity.

    - `Chunk` (class) in models.py
    - `make_chunk` (function) in conftest.py
    - `_CHUNK_SQL` (variable) in store.py
    - `fuse_scores` (function) in search.py
    - `fuse` (function) in fuse.py — prefix collision
    """
    seed_index(
        index,
        [
            make_chunk(
                "c_chunk_class",
                name="Chunk",
                kind=ChunkKind.CLASS,
                path="src/models.py",
            ),
            make_chunk(
                "c_make_chunk",
                name="make_chunk",
                kind=ChunkKind.FUNCTION,
                path="tests/conftest.py",
            ),
            make_chunk(
                "c_chunk_sql",
                name="_CHUNK_SQL",
                kind=ChunkKind.VARIABLE,
                path="src/store.py",
            ),
            make_chunk(
                "c_fuse_scores",
                name="fuse_scores",
                kind=ChunkKind.FUNCTION,
                path="src/search.py",
            ),
            make_chunk(
                "c_fuse",
                name="fuse",
                kind=ChunkKind.FUNCTION,
                path="src/fuse.py",
            ),
        ],
        head_ref,
    )
    return index


@pytest.mark.parametrize(
    ("pattern", "expected_names"),
    [
        # Tier 1: exact match
        ("Chunk", {"Chunk"}),
        ("fuse_scores", {"fuse_scores"}),
        ("make_chunk", {"make_chunk"}),
        # Tier 2: case-insensitive exact
        ("chunk", {"Chunk"}),
        ("FUSE_SCORES", {"fuse_scores"}),
        # Tier 3: prefix (no exact/iexact for "fuse_" exists)
        ("fuse_", {"fuse_scores"}),
        # Tier 4: substring (no higher tier matches "unk")
        ("unk", {"Chunk", "make_chunk", "_CHUNK_SQL"}),
    ],
    ids=[
        "exact-Chunk",
        "exact-fuse_scores",
        "exact-make_chunk",
        "iexact-chunk",
        "iexact-FUSE_SCORES",
        "prefix-fuse_",
        "substring-unk",
    ],
)
def test_name_matches_tiering(
    name_index: Index,
    pattern: str,
    expected_names: set[str],
) -> None:
    [repo] = name_index.repos()
    assert set(name_index.name_matches(pattern, within=[repo.at(COMMIT)])["name"]) == expected_names


def test_name_matches_across_projects_keep_the_best_tier(
    name_index: Index, head_ref: SnapshotRef
) -> None:
    """Each project's file returns its own best tier; the search keeps the best of those.

    The other project holds only a prefix match, which its file returns as
    its best; the first project's case-insensitive exact match outranks it.
    """
    other = name_index.register("/other").at(COMMIT)
    seed_index(name_index, [make_chunk("c_chunk_size", name="chunk_size", path="size.py")], other)

    within = [head_ref, other]
    assert set(name_index.name_matches("chunk", within=within)["name"]) == {"Chunk"}
