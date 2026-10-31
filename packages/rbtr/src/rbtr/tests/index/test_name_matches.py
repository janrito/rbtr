"""Tiered name resolution for name_matches.

name_matches resolves a user-supplied symbol name to chunks.
It should prefer exact matches, falling back through tiers:
exact → case-insensitive exact → prefix → substring.
Only the best tier that has matches should be returned.

A qualified name (`Class.method`, `module.Class.method`) resolves to
the symbol whose module and scope end with the qualifier.
"""

from __future__ import annotations

import pytest

from rbtr.config import config
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


@pytest.fixture
def qualified_index(index: Index, head_ref: SnapshotRef) -> Index:
    """Store whose symbols are told apart only by where they are.

    - `add_cli_args` in `EngineArgs` and in `AsyncEngineArgs`, two files
    - `m` in the nested scope `Outer::Inner`
    - the doc heading `Release 1.2`, whose own name holds a separator
    """
    seed_index(
        index,
        [
            make_chunk(
                "c_engine_add",
                name="add_cli_args",
                scope="EngineArgs",
                kind=ChunkKind.METHOD,
                path="pkg/engine/arg_utils.py",
                language="python",
            ),
            make_chunk(
                "c_async_add",
                name="add_cli_args",
                scope="AsyncEngineArgs",
                kind=ChunkKind.METHOD,
                path="pkg/engine/async_args.py",
                language="python",
            ),
            make_chunk(
                "c_inner_m",
                name="m",
                scope="Outer::Inner",
                kind=ChunkKind.METHOD,
                path="pkg/models.py",
                language="python",
            ),
            make_chunk(
                "c_release",
                name="Release 1.2",
                kind=ChunkKind.DOC_SECTION,
                path="CHANGES.md",
                language="markdown",
            ),
        ],
        head_ref,
    )
    return index


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("EngineArgs.add_cli_args", {("EngineArgs", "add_cli_args")}),
        ("EngineArgs.add_cli", {("EngineArgs", "add_cli_args")}),
        (
            "add_cli_args",
            {("EngineArgs", "add_cli_args"), ("AsyncEngineArgs", "add_cli_args")},
        ),
        ("Inner.m", {("Outer::Inner", "m")}),
        ("Outer.m", set()),
        ("Other.add_cli_args", set()),
        ("pkg.engine.arg_utils.EngineArgs.add_cli_args", {("EngineArgs", "add_cli_args")}),
        ("pkg.models.Inner.m", set()),
        ("Release 1.2", {("", "Release 1.2")}),
    ],
    ids=[
        "scope",
        "last-part-by-tier",
        "bare-name-finds-every-scope",
        "scope-suffix",
        "scope-gap-finds-nothing",
        "wrong-scope-does-not-fall-back",
        "module-then-scope",
        "after-a-module-the-scope-is-whole",
        "literal-name-with-separator",
    ],
)
def test_name_matches_qualified(
    qualified_index: Index,
    pattern: str,
    expected: set[tuple[str, str]],
) -> None:
    [repo] = qualified_index.repos()
    found = qualified_index.name_matches(pattern, within=[repo.at(COMMIT)])
    assert set(found.select("scope", "name").iter_rows()) == expected


@pytest.mark.parametrize(
    ("separators", "pattern", "expected"),
    [
        ([".", "#", "/"], "EngineArgs#add_cli_args", {("EngineArgs", "add_cli_args")}),
        (["."], "EngineArgs#add_cli_args", set()),
        (["."], "EngineArgs::add_cli_args", {("EngineArgs", "add_cli_args")}),
    ],
    ids=["configured-separator", "unconfigured-separator", "scope-separator-always"],
)
def test_qualifier_separators_decide_what_qualifies(
    qualified_index: Index,
    monkeypatch: pytest.MonkeyPatch,
    separators: list[str],
    pattern: str,
    expected: set[tuple[str, str]],
) -> None:
    monkeypatch.setattr(config, "qualifier_separators", separators)
    [repo] = qualified_index.repos()
    found = qualified_index.name_matches(pattern, within=[repo.at(COMMIT)])
    assert set(found.select("scope", "name").iter_rows()) == expected


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
