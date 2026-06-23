"""Tiered name resolution for match_by_name.

match_by_name resolves a user-supplied symbol name to chunks.
It should prefer exact matches, falling back through tiers:
exact → case-insensitive exact → prefix → substring.
Only the best tier that has matches should be returned.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import ChunkKind
from rbtr.index.store import IndexStore

from .conftest import make_chunk, seed_store

COMMIT = "head"


@pytest.fixture
def name_store(store: IndexStore) -> IndexStore:
    """Store with chunks that create name ambiguity.

    - `Chunk` (class) in models.py
    - `make_chunk` (function) in conftest.py
    - `_CHUNK_SQL` (variable) in store.py
    - `fuse_scores` (function) in search.py
    - `fuse` (function) in fuse.py — prefix collision
    """
    seed_store(
        store,
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
        commit_sha=COMMIT,
    )
    return store


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
def test_match_by_name_tiering(
    name_store: IndexStore,
    pattern: str,
    expected_names: set[str],
) -> None:
    results = name_store.match_by_name(COMMIT, pattern, repo_id=1)
    assert {c.name for c in results} == expected_names
