"""Tiered name resolution for name_matches.

name_matches resolves a user-supplied symbol name to chunks.
It should prefer exact matches, falling back through tiers:
exact → case-insensitive exact → prefix → substring.
Only the best tier that has matches should be returned.

A qualified name (`Class.method`, `module.Class.method`) resolves to
the symbol whose module and scope end with the qualifier.
"""

from __future__ import annotations

import pygit2
import pytest

from rbtr.config import config
from rbtr.domain.models import SnapshotRef
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.index import Index

from ..conftest import make_commit


@pytest.fixture(scope="module")
def name_ref(module_index: Index, module_repo: pygit2.Repository) -> SnapshotRef:
    """A snapshot, built into `module_index`, whose symbols create name ambiguity.

    - `Chunk` (class) in models.py
    - `make_chunk` (function) in conftest.py
    - `_CHUNK_SQL` (variable) in store.py
    - `fuse_scores` (function) in search.py
    - `fuse` (function) in fuse.py — prefix collision
    """
    models = b"""\
class Chunk:
    pass
"""
    conftest = b"""\
def make_chunk():
    pass
"""
    store = b"""\
_CHUNK_SQL = "SELECT 1"
"""
    search = b"""\
def fuse_scores():
    pass
"""
    fuse = b"""\
def fuse():
    pass
"""
    files = {
        "src/models.py": models,
        "tests/conftest.py": conftest,
        "src/store.py": store,
        "src/search.py": search,
        "src/fuse.py": fuse,
    }
    sha = str(make_commit(module_repo, files, ref="refs/heads/names"))
    path = normalise_repo_path(module_repo.workdir)
    build_index(path, sha, module_index)
    return module_index.register(path).at(sha)


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
    module_index: Index,
    name_ref: SnapshotRef,
    pattern: str,
    expected_names: set[str],
) -> None:
    assert set(module_index.name_matches(pattern, within=[name_ref])["name"]) == expected_names


@pytest.fixture(scope="module")
def qualified_ref(module_index: Index, module_repo: pygit2.Repository) -> SnapshotRef:
    """A snapshot, built into `module_index`, whose symbols are told apart only by where they are.

    - `add_cli_args` in `EngineArgs` and in `AsyncEngineArgs`, two files
    - `m` in the nested scope `Outer::Inner`
    - the doc heading `Release 1.2`, whose own name holds a separator
    """
    arg_utils = b"""\
class EngineArgs:
    def add_cli_args(self):
        pass
"""
    async_args = b"""\
class AsyncEngineArgs:
    def add_cli_args(self):
        pass
"""
    models = b"""\
class Outer:
    class Inner:
        def m(self):
            pass
"""
    changes = b"""\
# Release 1.2

Notes.
"""
    files = {
        "pkg/engine/arg_utils.py": arg_utils,
        "pkg/engine/async_args.py": async_args,
        "pkg/models.py": models,
        "CHANGES.md": changes,
    }
    sha = str(make_commit(module_repo, files, ref="refs/heads/qualified"))
    path = normalise_repo_path(module_repo.workdir)
    build_index(path, sha, module_index)
    return module_index.register(path).at(sha)


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
    module_index: Index,
    qualified_ref: SnapshotRef,
    pattern: str,
    expected: set[tuple[str, str]],
) -> None:
    found = module_index.name_matches(pattern, within=[qualified_ref])
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
    module_index: Index,
    qualified_ref: SnapshotRef,
    monkeypatch: pytest.MonkeyPatch,
    separators: list[str],
    pattern: str,
    expected: set[tuple[str, str]],
) -> None:
    monkeypatch.setattr(config, "qualifier_separators", separators)
    found = module_index.name_matches(pattern, within=[qualified_ref])
    assert set(found.select("scope", "name").iter_rows()) == expected


@pytest.fixture(scope="module")
def other_project_ref(module_index: Index, tmp_path_factory: pytest.TempPathFactory) -> SnapshotRef:
    """Another project, built into `module_index`, holding only `chunk_size`."""
    size = b"""\
def chunk_size():
    pass
"""
    repo = pygit2.init_repository(str(tmp_path_factory.mktemp("other-project")), bare=False)
    sha = str(make_commit(repo, {"size.py": size}, ref="refs/heads/main"))
    path = normalise_repo_path(repo.workdir)
    build_index(path, sha, module_index)
    return module_index.register(path).at(sha)


def test_name_matches_across_projects_keep_the_best_tier(
    module_index: Index, name_ref: SnapshotRef, other_project_ref: SnapshotRef
) -> None:
    """Each project's file returns its own best tier; the search keeps the best of those.

    The other project holds only a prefix match, which its file returns as
    its best; the first project's case-insensitive exact match outranks it.
    """
    within = [name_ref, other_project_ref]
    assert set(module_index.name_matches("chunk", within=within)["name"]) == {"Chunk"}
