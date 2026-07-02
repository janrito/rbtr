"""Tests for the index orchestrator — build mechanics."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import replace
from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.index.models import ChunkKind, EdgeKind, IndexResult, Snapshot
from rbtr.index.orchestrator import build_index
from rbtr.index.store import IndexStore
from rbtr.index.treesitter import _get_query
from rbtr.languages import get_manager


@pytest.fixture(scope="module")
def built_index(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[tuple[IndexStore, IndexResult, str]]:
    """Pre-built index over the standard git_repo shape.

    Module-scoped: built once, shared read-only by tests that
    assert on index structure.
    """
    tmp = tmp_path_factory.mktemp("built")
    repo = pygit2.init_repository(str(tmp / "repo"), bare=False, initial_head="main")
    files = {
        "src/models.py": b'"""Data models."""\n\nclass User:\n    pass\n\nclass Order:\n    pass\n',
        "src/utils.py": b'"""Utility functions."""\n\ndef helper():\n    return 42\n\ndef format_name(name):\n    return name.strip()\n',
        "src/main.py": b'"""Main module."""\n\nfrom src.models import User\nfrom src.utils import helper\n\ndef run():\n    u = User()\n    return helper()\n',
        "tests/test_utils.py": b'"""Tests for utils."""\n\nfrom src.utils import helper, format_name\n\ndef test_helper():\n    assert helper() == 42\n\ndef test_format():\n    assert format_name("  hi  ") == "hi"\n',
        "README.md": b"# My Project\n\nThis project uses `helper` and `User` for things.\n\n## Setup\n\nRun `format_name` to clean strings.\n",
    }
    repo_root = tmp / "repo"
    index = repo.index
    for path, content in files.items():
        full = repo_root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(content)
        index.add(path)
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Initial commit", tree_oid, [])
    sha = str(repo.head.target)

    store = IndexStore(writable=True)
    result = build_index(repo.workdir, sha, store, repo_id=1)
    yield store, result, sha
    store.close()


@pytest.fixture
def monorepo_repo(tmp_path: Path) -> pygit2.Repository:
    """Git repo with a packages/*/src monorepo layout.

    The importer uses an absolute dotted path (`core.models`) to a file below
    an unknown root, which only the suffix-matching tier resolves.
    """
    repo = pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")
    files = {
        "packages/core/src/core/models.py": b'''\
"""Core models."""


class Widget:
    pass
''',
        "packages/app/src/app/main.py": b'''\
"""App entry point."""

from core.models import Widget


def run() -> Widget:
    return Widget()
''',
    }
    index = repo.index
    for path, content in files.items():
        full = tmp_path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(content)
        index.add(path)
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Initial commit", tree_oid, [])
    return repo


def test_build_index_resolves_monorepo_absolute_import(
    store: IndexStore, monorepo_repo: pygit2.Repository
) -> None:
    """An absolute import across packages/*/src yields a retrievable edge.

    Mirrors the path handle_find_refs walks: resolve the target chunk, then
    query edges by target_id.
    """
    sha = str(monorepo_repo.head.target)
    build_index(monorepo_repo.workdir, sha, store, repo_id=1)
    widget = next(
        c
        for c in store.get_chunks(sha, repo_id=1)
        if c.name == "Widget"
        and c.kind == ChunkKind.CLASS
        and c.file_path == "packages/core/src/core/models.py"
    )

    edges = store.get_edges(sha, target_id=widget.id, repo_id=1)
    assert any(e.kind == EdgeKind.IMPORTS for e in edges), (
        "absolute import across packages/*/src produced no inbound edge"
    )


def test_build_index_creates_chunks(
    built_index: tuple[IndexStore, IndexResult, str],
) -> None:
    _store, result, _sha = built_index
    assert isinstance(result, IndexResult)
    assert result.stats.total_files > 0
    assert result.stats.total_chunks > 0
    assert not result.errors


def test_build_index_creates_snapshots(
    built_index: tuple[IndexStore, IndexResult, str],
) -> None:
    store, _result, sha = built_index
    chunks = store.get_chunks(sha, repo_id=1)
    assert len(chunks) > 0
    file_paths = {c.file_path for c in chunks}
    assert "src/models.py" in file_paths
    assert "src/utils.py" in file_paths


def test_build_index_extracts_symbols(
    built_index: tuple[IndexStore, IndexResult, str],
) -> None:
    store, _result, sha = built_index
    chunks = store.get_chunks(sha, repo_id=1)
    names = {c.name for c in chunks}
    assert "User" in names
    assert "Order" in names
    assert "helper" in names
    assert "run" in names


def test_build_index_creates_edges(
    built_index: tuple[IndexStore, IndexResult, str],
) -> None:
    store, _result, sha = built_index
    edges = store.get_edges(sha, repo_id=1)
    assert len(edges) > 0
    edge_kinds = {e.kind for e in edges}
    assert EdgeKind.IMPORTS in edge_kinds


def test_build_index_markdown_chunking(
    built_index: tuple[IndexStore, IndexResult, str],
) -> None:
    store, _result, sha = built_index
    chunks = store.get_chunks(sha, file_path="README.md", repo_id=1)
    assert len(chunks) > 0
    kinds = {c.kind for c in chunks}
    assert ChunkKind.DOC_SECTION in kinds


def test_build_index_fts_available(
    built_index: tuple[IndexStore, IndexResult, str],
) -> None:
    store, _result, sha = built_index
    results = store.match_fulltext(sha, "helper", repo_id=1)
    assert len(results) > 0


def test_build_index_progress_callback(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    calls = []
    build_index(
        git_repo.workdir,
        commit_sha,
        store,
        repo_id=1,
        on_progress=lambda p, d, t: calls.append((p, d, t)),
    )

    assert len(calls) > 0
    # Last call should have done == total.
    _phase, done, total = calls[-1]
    assert done == total


def test_build_index_cache_hit(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Second build should hit the cache for all files."""
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    r2 = build_index(git_repo.workdir, commit_sha, store, repo_id=1)

    assert r2.stats.skipped_files == r2.stats.total_files
    assert r2.stats.parsed_files == 0


@pytest.fixture
def linked_worktree(
    git_repo: pygit2.Repository, tmp_path_factory: pytest.TempPathFactory
) -> pygit2.Repository:
    """A linked git worktree of `git_repo` at the same HEAD commit.

    Models a second checkout of one repository: a distinct workdir over
    the *same* object store, so identical content yields identical blob
    SHAs — the real-world worktree/clone case the content-addressed
    store deduplicates.
    """
    path = tmp_path_factory.mktemp("linked") / "checkout"
    git_repo.add_worktree("checkout", str(path))
    return pygit2.Repository(pygit2.discover_repository(str(path)))


def test_build_dedups_across_linked_worktrees(
    git_repo: pygit2.Repository,
    linked_worktree: pygit2.Repository,
    store: IndexStore,
    commit_sha: str,
) -> None:
    """Indexing two real checkouts of one repo reuses chunks and edges.

    Build the main checkout (repo 1) then the linked worktree (repo 2)
    at the same commit. Because their blobs are identical, repo 2 must
    reuse every chunk — nothing re-parsed — see the same chunk ids, and
    have the same edges inferred (from the shared chunks read back via
    snapshots) even though extraction was skipped.
    """
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    wt = build_index(linked_worktree.workdir, commit_sha, store, repo_id=2)

    ids_1 = sorted(c.id for c in store.get_chunks(commit_sha, repo_id=1))
    ids_2 = sorted(c.id for c in store.get_chunks(commit_sha, repo_id=2))
    assert ids_1
    assert ids_1 == ids_2

    # Second checkout reuses chunks — nothing re-parsed.
    assert wt.stats.parsed_files == 0
    assert wt.stats.skipped_files == wt.stats.total_files

    # Edges inferred for repo 2 from the shared chunks, matching repo 1.
    edges_1 = {(e.source_id, e.target_id, e.kind) for e in store.get_edges(commit_sha, repo_id=1)}
    edges_2 = {(e.source_id, e.target_id, e.kind) for e in store.get_edges(commit_sha, repo_id=2)}
    assert edges_1
    assert edges_1 == edges_2


@pytest.fixture
def divergent_sha(linked_worktree: pygit2.Repository) -> str:
    """Advance the linked worktree by one commit editing `src/utils.py`.

    Makes the worktree diverge from the main checkout: most files stay
    byte-identical (shared blobs), but `src/utils.py` differs — so its
    chunks must be distinct per checkout while the rest deduplicate.
    Returns the diverging commit's SHA; the worktree itself is the
    `linked_worktree` fixture.
    """
    utils = Path(linked_worktree.workdir) / "src" / "utils.py"
    utils.write_bytes(b'"""Utility functions."""\n\ndef helper():\n    return 99\n')
    linked_worktree.index.add("src/utils.py")
    linked_worktree.index.write()
    tree_oid = linked_worktree.index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    linked_worktree.create_commit(
        "HEAD", sig, sig, "Diverge utils", tree_oid, [linked_worktree.head.target]
    )
    return str(linked_worktree.head.target)


def test_build_dedups_shared_files_across_divergent_worktrees(
    git_repo: pygit2.Repository,
    linked_worktree: pygit2.Repository,
    divergent_sha: str,
    store: IndexStore,
    commit_sha: str,
) -> None:
    """Worktrees on different branches dedup shared files; diverged file differs.

    Build the main checkout (repo 1) at its commit and the worktree
    (repo 2) at its diverged commit. Unchanged files share chunk ids
    across both (one physical row); the edited `src/utils.py` yields a
    disjoint chunk set per checkout, with neither repo seeing the
    other's version.
    """
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    build_index(linked_worktree.workdir, divergent_sha, store, repo_id=2)

    # Unchanged file: same chunk ids in both checkouts (deduped).
    models_1 = sorted(
        c.id for c in store.get_chunks(commit_sha, file_path="src/models.py", repo_id=1)
    )
    models_2 = sorted(
        c.id for c in store.get_chunks(divergent_sha, file_path="src/models.py", repo_id=2)
    )
    assert models_1
    assert models_1 == models_2

    # Diverged file: disjoint chunk sets, neither repo sees the other's.
    utils_1 = {c.id for c in store.get_chunks(commit_sha, file_path="src/utils.py", repo_id=1)}
    utils_2 = {c.id for c in store.get_chunks(divergent_sha, file_path="src/utils.py", repo_id=2)}
    assert utils_1
    assert utils_2
    assert utils_1.isdisjoint(utils_2)


def test_build_index_idempotent_edges(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Re-building should not duplicate edges."""
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    e1 = store.get_edges(commit_sha, repo_id=1)

    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    e2 = store.get_edges(commit_sha, repo_id=1)

    assert len(e1) == len(e2)


def test_build_index_replaces_snapshots_for_same_ref(
    git_repo: pygit2.Repository, store: IndexStore, tmp_path: Path
) -> None:
    """Re-indexing the same ref removes files deleted since older reviews."""
    head = git_repo.head.peel(pygit2.Commit)
    git_repo.branches.local.create("review-head", head)
    git_repo.set_head("refs/heads/review-head")
    git_repo.checkout_head(strategy=pygit2.GIT_CHECKOUT_FORCE)  # type: ignore[no-untyped-call]  # pygit2 untyped

    # First index pass includes src/main.py.
    build_index(git_repo.workdir, "review-head", store, repo_id=1)
    before_paths = {c.file_path for c in store.get_chunks("review-head", repo_id=1)}
    assert "src/main.py" in before_paths

    # Delete src/main.py and commit on the same ref name.
    main_path = tmp_path / "src" / "main.py"
    main_path.unlink()
    index = git_repo.index
    index.remove("src/main.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Remove main", tree_oid, [parent.id])

    # Second pass at the same ref should not leak deleted-file chunks.
    build_index(git_repo.workdir, "review-head", store, repo_id=1)
    after_paths = {c.file_path for c in store.get_chunks("review-head", repo_id=1)}
    assert "src/main.py" not in after_paths
    assert "src/utils.py" in after_paths


def test_build_index_metadata_round_trip(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Import metadata should survive store round-trip."""
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)

    chunks = store.get_chunks(commit_sha, file_path="src/main.py", repo_id=1)
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) > 0

    # At least one import should have structural metadata.
    has_metadata = any(c.metadata.module or c.metadata.names for c in imports)
    assert has_metadata


def test_build_index_marks_commit_indexed(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Successful build_index records a completion row."""
    assert store.has_indexed(1, commit_sha) is False
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    assert store.has_indexed(1, commit_sha) is True


def test_build_index_sweeps_residue_from_crashed_builds(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """A successful build cleans up dangling snapshots/edges from prior crashes."""
    # Simulate residue: a snapshot for a commit that never finished.
    with store.session() as ws:
        ws.insert_snapshots(
            [Snapshot(commit_sha="crashed_sha", file_path="leftover.py", blob_sha="leftover_blob")],
            repo_id=1,
        )
    assert store.has_indexed(1, "crashed_sha") is False

    build_index(git_repo.workdir, commit_sha, store, repo_id=1)

    # The legit commit is indexed; the crashed residue is gone.
    assert store.has_indexed(1, commit_sha) is True
    assert store.get_chunks("crashed_sha", repo_id=1) == []


def test_build_index_empty_repo(tmp_path: Path, store: IndexStore) -> None:
    """Index an empty repo (no files)."""
    repo = pygit2.init_repository(str(tmp_path / "empty"), bare=False, initial_head="main")

    index = repo.index
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Empty", tree_oid, [])

    sha = str(repo.head.target)
    result = build_index(repo.workdir, sha, store, repo_id=1)

    assert result.stats.total_files == 0
    assert result.stats.total_chunks == 0
    assert not result.errors


def test_query_cache_produces_identical_chunks(tmp_path: Path) -> None:
    """Cached tree-sitter queries must produce the same chunks as fresh ones.

    `_get_query()` caches compiled `Query` objects per language.
    This test creates many same-language files, builds twice (cold cache
    vs warm cache), and asserts identical extraction results.
    """
    repo = pygit2.init_repository(str(tmp_path / "cache_test"), bare=False, initial_head="main")

    # Create 10 Python files — each with a uniquely named function.
    for i in range(10):
        path = tmp_path / "cache_test" / f"mod_{i}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def unique_func_{i}():\n    return {i}\n")

    index = repo.index
    for i in range(10):
        index.add(f"mod_{i}.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])

    sha = str(repo.head.target)

    # Clear the query cache to ensure a cold start.

    _get_query.cache_clear()

    store1 = IndexStore(writable=True)
    store2 = IndexStore(writable=True)
    try:
        build_index(repo.workdir, sha, store1, repo_id=1)
        chunks1 = {c.name for c in store1.get_chunks(sha, repo_id=1)}

        # Second build — queries are now cached.
        _get_query.cache_clear()
        build_index(repo.workdir, sha, store2, repo_id=1)
        chunks2 = {c.name for c in store2.get_chunks(sha, repo_id=1)}

        # All 10 functions must be present in both.
        for i in range(10):
            assert f"unique_func_{i}" in chunks1, f"Cold cache missing unique_func_{i}"
            assert f"unique_func_{i}" in chunks2, f"Warm cache missing unique_func_{i}"

        assert chunks1 == chunks2
    finally:
        store1.close()
        store2.close()


def test_build_rebuilds_fts_at_commit(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """FTS is rebuilt at the end of build_index, not on first search."""
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)

    # match_fulltext finds results — the build rebuilt FTS.
    results = store.match_fulltext(commit_sha, "helper", repo_id=1)
    assert len(results) > 0


def test_build_index_unknown_language(tmp_path: Path, store: IndexStore) -> None:
    """Files with unknown extensions should get plaintext chunking."""
    repo = pygit2.init_repository(str(tmp_path / "unknown"), bare=False, initial_head="main")

    (tmp_path / "unknown" / "data.xyz").write_text("""\
line1
line2
line3
""")
    index = repo.index
    index.add("data.xyz")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Add unknown file", tree_oid, [])

    sha = str(repo.head.target)
    result = build_index(repo.workdir, sha, store, repo_id=1)

    assert result.stats.total_files == 1
    chunks = store.get_chunks(sha, repo_id=1)
    assert len(chunks) > 0
    assert all(c.kind == ChunkKind.RAW_CHUNK for c in chunks)


def test_build_index_prose_txt_detected(tmp_path: Path, store: IndexStore) -> None:
    """A .txt file with Markdown/RST content gets DOC_SECTION chunks."""
    repo = pygit2.init_repository(str(tmp_path / "prose"), bare=False, initial_head="main")

    (tmp_path / "prose" / "CHANGES.txt").write_text("""\
Changelog
=========

Version 1.0
-----------

Initial release.
""")
    index = repo.index
    index.add("CHANGES.txt")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Add changelog", tree_oid, [])

    sha = str(repo.head.target)
    build_index(repo.workdir, sha, store, repo_id=1)

    chunks = store.get_chunks(sha, repo_id=1)
    assert len(chunks) > 0
    assert any(c.language == "rst" for c in chunks)
    assert any(c.kind == ChunkKind.DOC_SECTION for c in chunks)


def test_build_index_prose_blob_dedup(tmp_path: Path, store: IndexStore) -> None:
    """Prose-detected blobs are deduped on rebuild (has_blob prose fallback)."""
    repo = pygit2.init_repository(str(tmp_path / "prose_dedup"), bare=False, initial_head="main")

    (tmp_path / "prose_dedup" / "README").write_text("""\
# My Project

Description here.
""")
    index = repo.index
    index.add("README")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])

    sha = str(repo.head.target)
    build_index(repo.workdir, sha, store, repo_id=1)
    r2 = build_index(repo.workdir, sha, store, repo_id=1)

    assert r2.stats.skipped_files == r2.stats.total_files
    assert r2.stats.parsed_files == 0


def test_build_index_version_gated_reextraction(
    tmp_path: Path, store: IndexStore, mocker: MockerFixture
) -> None:
    """Bumping `language_plugin_version` forces re-extraction."""
    repo = pygit2.init_repository(str(tmp_path / "ver"), bare=False, initial_head="main")

    (tmp_path / "ver" / "doc.md").write_text("# Hello\n\nWorld.\n")
    index = repo.index
    index.add("doc.md")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])

    sha = str(repo.head.target)
    r1 = build_index(repo.workdir, sha, store, repo_id=1)
    assert r1.stats.parsed_files >= 1

    # Bump the markdown registration's version.
    mgr = get_manager()
    orig_reg = mgr.get_registration("markdown")
    assert orig_reg is not None
    bumped = replace(orig_reg, language_plugin_version=99)
    mocker.patch.object(
        mgr, "get_registration", side_effect=lambda lid: bumped if lid == "markdown" else orig_reg
    )

    r2 = build_index(repo.workdir, sha, store, repo_id=1)
    assert r2.stats.parsed_files >= 1, "version bump should force re-extraction"


@pytest.fixture
def svelte_repo(tmp_path: Path) -> pygit2.Repository:
    """Git repo containing the committed `samples/svelte/` project.

    Uses the real sample so these tests stay in step with the extraction
    they document.
    """
    sample = Path(__file__).parents[1] / "languages" / "samples" / "svelte"
    repo = pygit2.init_repository(str(tmp_path / "sfc"), bare=False, initial_head="main")
    index = repo.index
    for src in sorted(sample.glob("*")):
        (tmp_path / "sfc" / src.name).write_bytes(src.read_bytes())
        index.add(src.name)
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])
    return repo


def test_build_index_dedups_sfc(svelte_repo: pygit2.Repository, store: IndexStore) -> None:
    """A multi-language SFC is skipped on rebuild, not re-parsed every build.

    Also guards per-language versioning: a delegated chunk stamped with the
    wrong (host) version would miss the version-map gate and re-extract here.
    """
    sha = str(svelte_repo.head.target)
    build_index(svelte_repo.workdir, sha, store, repo_id=1)
    rebuild = build_index(svelte_repo.workdir, sha, store, repo_id=1)

    assert rebuild.stats.skipped_files == rebuild.stats.total_files
    assert rebuild.stats.parsed_files == 0


@pytest.mark.parametrize("plugin", ["typescript", "svelte"])
def test_build_index_reextracts_sfc_on_plugin_bump(
    svelte_repo: pygit2.Repository, store: IndexStore, mocker: MockerFixture, plugin: str
) -> None:
    """Bumping any contributor — the embedded language or the host — re-extracts.

    Confirms the version map gates on every language in the file, not just one.
    """
    sha = str(svelte_repo.head.target)
    build_index(svelte_repo.workdir, sha, store, repo_id=1)

    mgr = get_manager()
    target = mgr.get_registration(plugin)
    assert target is not None
    bumped = replace(target, language_plugin_version=target.language_plugin_version + 1)
    originals = {lid: mgr.get_registration(lid) for lid in mgr.all_language_ids()}
    mocker.patch.object(
        mgr,
        "get_registration",
        side_effect=lambda lid: bumped if lid == plugin else originals.get(lid),
    )

    rebuild = build_index(svelte_repo.workdir, sha, store, repo_id=1)
    assert rebuild.stats.parsed_files >= 1


def test_build_index_dedups_empty_file(tmp_path: Path, store: IndexStore) -> None:
    """An empty file is skipped on rebuild, not re-parsed every time.

    An empty `__init__.py` produces no definition chunks; the host-presence
    chunk records its language so `has_blob` hits on the second build.
    """
    repo = pygit2.init_repository(str(tmp_path / "empty"), bare=False, initial_head="main")
    (tmp_path / "empty" / "pkg").mkdir(parents=True)
    (tmp_path / "empty" / "pkg" / "__init__.py").write_text("")
    index = repo.index
    index.add("pkg/__init__.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])
    sha = str(repo.head.target)

    build_index(repo.workdir, sha, store, repo_id=1)
    r2 = build_index(repo.workdir, sha, store, repo_id=1)

    assert r2.stats.skipped_files == r2.stats.total_files
    assert r2.stats.parsed_files == 0


def test_build_index_chunk_ids_stable(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Build twice, compare chunk ID sets — no phantom inserts or deletes."""
    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    ids_1 = {c.id for c in store.get_chunks(commit_sha, repo_id=1)}
    assert len(ids_1) > 0

    build_index(git_repo.workdir, commit_sha, store, repo_id=1)
    ids_2 = {c.id for c in store.get_chunks(commit_sha, repo_id=1)}

    assert ids_1 == ids_2
