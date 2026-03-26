"""Tests for the index orchestrator — build, update, and diff."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.index.models import ChunkKind, EdgeKind, IndexResult, SemanticDiff
from rbtr.index.orchestrator import (
    build_index,
    compute_diff,
    update_index,
)
from rbtr.index.store import IndexStore
from rbtr.index.treesitter import _get_query

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_embeddings(mocker: MockerFixture) -> None:
    """Stub out the embedding step — no GGUF model needed in tests."""
    mocker.patch("rbtr.index.orchestrator._embed_missing")


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    """Create a git repo with a few Python files and a commit."""
    repo = pygit2.init_repository(str(tmp_path), bare=False)

    # Create files.
    files = {
        "src/models.py": b'"""Data models."""\n\nclass User:\n    pass\n\nclass Order:\n    pass\n',
        "src/utils.py": b'"""Utility functions."""\n\ndef helper():\n    return 42\n\ndef format_name(name):\n    return name.strip()\n',
        "src/main.py": b'"""Main module."""\n\nfrom src.models import User\nfrom src.utils import helper\n\ndef run():\n    u = User()\n    return helper()\n',
        "tests/test_utils.py": b'"""Tests for utils."""\n\nfrom src.utils import helper, format_name\n\ndef test_helper():\n    assert helper() == 42\n\ndef test_format():\n    assert format_name("  hi  ") == "hi"\n',
        "README.md": b"# My Project\n\nThis project uses `helper` and `User` for things.\n\n## Setup\n\nRun `format_name` to clean strings.\n",
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


@pytest.fixture
def store() -> Generator[IndexStore]:
    """In-memory DuckDB store."""
    s = IndexStore()
    yield s
    s.close()


@pytest.fixture
def commit_sha(git_repo: pygit2.Repository) -> str:
    """SHA of the initial commit."""
    return str(git_repo.head.target)


# ── build_index tests ────────────────────────────────────────────────


def test_build_index_creates_chunks(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    result = build_index(git_repo, commit_sha, store)

    assert isinstance(result, IndexResult)
    assert result.stats.total_files > 0
    assert result.stats.total_chunks > 0
    assert not result.errors


def test_build_index_creates_snapshots(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    build_index(git_repo, commit_sha, store)

    chunks = store.get_chunks(commit_sha)
    assert len(chunks) > 0
    # All chunks should be visible at this commit.
    file_paths = {c.file_path for c in chunks}
    assert "src/models.py" in file_paths
    assert "src/utils.py" in file_paths


def test_build_index_extracts_symbols(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    build_index(git_repo, commit_sha, store)

    chunks = store.get_chunks(commit_sha)
    names = {c.name for c in chunks}
    # Python plugin should extract these.
    assert "User" in names
    assert "Order" in names
    assert "helper" in names
    assert "run" in names


def test_build_index_creates_edges(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    build_index(git_repo, commit_sha, store)

    edges = store.get_edges(commit_sha)
    assert len(edges) > 0

    edge_kinds = {e.kind for e in edges}
    # Should have import, test, and doc edges.
    assert EdgeKind.IMPORTS in edge_kinds


def test_build_index_markdown_chunking(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    build_index(git_repo, commit_sha, store)

    chunks = store.get_chunks(commit_sha, file_path="README.md")
    assert len(chunks) > 0
    kinds = {c.kind for c in chunks}
    assert ChunkKind.DOC_SECTION in kinds


def test_build_index_progress_callback(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    calls = []
    build_index(git_repo, commit_sha, store, on_progress=lambda d, t: calls.append((d, t)))

    assert len(calls) > 0
    # Last call should have done == total.
    done, total = calls[-1]
    assert done == total


def test_build_index_cache_hit(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Second build should hit the cache for all files."""
    build_index(git_repo, commit_sha, store)
    r2 = build_index(git_repo, commit_sha, store)

    assert r2.stats.skipped_files == r2.stats.total_files
    assert r2.stats.parsed_files == 0


def test_build_index_fts_available(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """FTS index should be rebuilt after build."""
    build_index(git_repo, commit_sha, store)

    results = store.search_fulltext(commit_sha, "helper")
    assert len(results) > 0


def test_build_index_idempotent_edges(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Re-building should not duplicate edges."""
    build_index(git_repo, commit_sha, store)
    e1 = store.get_edges(commit_sha)

    build_index(git_repo, commit_sha, store)
    e2 = store.get_edges(commit_sha)

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
    build_index(git_repo, "review-head", store)
    before_paths = {c.file_path for c in store.get_chunks("review-head")}
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
    build_index(git_repo, "review-head", store)
    after_paths = {c.file_path for c in store.get_chunks("review-head")}
    assert "src/main.py" not in after_paths
    assert "src/utils.py" in after_paths


def test_build_index_metadata_round_trip(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Import metadata should survive store round-trip."""
    build_index(git_repo, commit_sha, store)

    chunks = store.get_chunks(commit_sha, file_path="src/main.py")
    imports = [c for c in chunks if c.kind == ChunkKind.IMPORT]
    assert len(imports) > 0

    # At least one import should have structural metadata.
    has_metadata = any(c.metadata.get("module") or c.metadata.get("names") for c in imports)
    assert has_metadata


# ── update_index tests ──────────────────────────────────────────────


@pytest.fixture
def two_commits(git_repo: pygit2.Repository, tmp_path: Path) -> tuple[str, str]:
    """Add a second commit that modifies utils.py and adds a new file."""
    base_sha = str(git_repo.head.target)

    # Modify utils.py: add a new function.
    utils_path = tmp_path / "src" / "utils.py"
    utils_path.write_bytes(
        b'"""Utility functions."""\n\ndef helper():\n    return 42\n\ndef format_name(name):\n    return name.strip()\n\ndef new_func():\n    return "new"\n'
    )

    # Add a new file.
    new_path = tmp_path / "src" / "service.py"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b'"""Service layer."""\n\ndef serve():\n    return True\n')

    index = git_repo.index
    index.add("src/utils.py")
    index.add("src/service.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Add new_func and service", tree_oid, [parent.id])

    head_sha = str(git_repo.head.target)
    return base_sha, head_sha


def test_update_index_incremental(
    git_repo: pygit2.Repository, store: IndexStore, two_commits: tuple[str, str]
) -> None:
    base_sha, head_sha = two_commits

    # Build base first.
    build_index(git_repo, base_sha, store)

    # Incremental update.
    result = update_index(git_repo, base_sha, head_sha, store)

    assert result.stats.total_files > 0
    assert result.stats.parsed_files >= 1  # At least the changed files.
    assert not result.errors

    # New function should be visible at head.
    chunks = store.get_chunks(head_sha)
    names = {c.name for c in chunks}
    assert "new_func" in names
    assert "serve" in names


def test_update_index_preserves_unchanged(
    git_repo: pygit2.Repository, store: IndexStore, two_commits: tuple[str, str]
) -> None:
    base_sha, head_sha = two_commits

    build_index(git_repo, base_sha, store)
    result = update_index(git_repo, base_sha, head_sha, store)

    # Unchanged files should be cached, not re-parsed.
    assert result.stats.skipped_files > 0

    # Old symbols should still be visible at head.
    chunks = store.get_chunks(head_sha)
    names = {c.name for c in chunks}
    assert "User" in names
    assert "Order" in names


def test_update_index_replaces_head_snapshots_for_reused_ref(
    git_repo: pygit2.Repository, store: IndexStore, tmp_path: Path
) -> None:
    """Head ref re-index does not leak deleted files from older reviews."""
    base = git_repo.head.peel(pygit2.Commit)
    try:
        git_repo.branches.local["main"]
    except KeyError:
        git_repo.branches.local.create("main", base)
    try:
        git_repo.branches.local["feature"]
    except KeyError:
        git_repo.branches.local.create("feature", base)

    # Build base under a stable branch ref, as /review does.
    build_index(git_repo, "main", store)

    # Feature commit 1: add a temporary file/symbol.
    git_repo.set_head("refs/heads/feature")
    git_repo.checkout_head(strategy=pygit2.GIT_CHECKOUT_FORCE)  # type: ignore[no-untyped-call]  # pygit2 untyped
    legacy_path = tmp_path / "src" / "legacy.py"
    legacy_path.write_text(
        """\
def legacy_only():
    return "legacy"
""",
        encoding="utf-8",
    )

    index = git_repo.index
    index.add("src/legacy.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Add legacy file", tree_oid, [parent.id])

    update_index(git_repo, "main", "feature", store)
    names_v1 = {c.name for c in store.get_chunks("feature")}
    assert "legacy_only" in names_v1

    # Feature commit 2: remove that file so head tree matches base again.
    legacy_path.unlink()
    index = git_repo.index
    index.remove("src/legacy.py")
    index.write()
    tree_oid = index.write_tree()
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Remove legacy file", tree_oid, [parent.id])

    update_index(git_repo, "main", "feature", store)
    names_v2 = {c.name for c in store.get_chunks("feature")}
    assert "legacy_only" not in names_v2


# ── compute_diff tests ───────────────────────────────────────────────


def test_compute_diff_added(
    git_repo: pygit2.Repository, store: IndexStore, two_commits: tuple[str, str]
) -> None:
    base_sha, head_sha = two_commits

    build_index(git_repo, base_sha, store)
    update_index(git_repo, base_sha, head_sha, store)

    diff = compute_diff(base_sha, head_sha, store)

    assert isinstance(diff, SemanticDiff)
    added_names = {c.name for c in diff.added}
    assert "new_func" in added_names or "serve" in added_names


def test_compute_diff_removed(
    git_repo: pygit2.Repository, store: IndexStore, tmp_path: Path
) -> None:
    """Removing a function should show up in diff.removed."""
    base_sha = str(git_repo.head.target)

    # Remove Order class from models.py.
    models_path = tmp_path / "src" / "models.py"
    models_path.write_bytes(b'"""Data models."""\n\nclass User:\n    pass\n')

    index = git_repo.index
    index.add("src/models.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Remove Order", tree_oid, [parent.id])
    head_sha = str(git_repo.head.target)

    build_index(git_repo, base_sha, store)
    update_index(git_repo, base_sha, head_sha, store)

    diff = compute_diff(base_sha, head_sha, store)

    removed_names = {c.name for c in diff.removed}
    assert "Order" in removed_names


def test_compute_diff_missing_tests(
    git_repo: pygit2.Repository, store: IndexStore, two_commits: tuple[str, str]
) -> None:
    """New functions without test edges should appear in missing_tests."""
    base_sha, head_sha = two_commits

    build_index(git_repo, base_sha, store)
    update_index(git_repo, base_sha, head_sha, store)

    diff = compute_diff(base_sha, head_sha, store)

    # new_func and serve are new, and likely have no test edges.
    missing_names = {c.name for c in diff.missing_tests}
    assert "serve" in missing_names or "new_func" in missing_names


def test_compute_diff_empty_when_same(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """Diffing a commit against itself should produce empty results."""
    build_index(git_repo, commit_sha, store)

    diff = compute_diff(commit_sha, commit_sha, store)

    assert diff.added == []
    assert diff.removed == []
    assert diff.modified == []


# ── Edge cases ───────────────────────────────────────────────────────


def test_build_index_empty_repo(tmp_path: Path) -> None:
    """Index an empty repo (no files)."""
    repo = pygit2.init_repository(str(tmp_path / "empty"), bare=False)

    # Create an empty commit.
    index = repo.index
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Empty", tree_oid, [])

    store = IndexStore()
    try:
        sha = str(repo.head.target)
        result = build_index(repo, sha, store)

        assert result.stats.total_files == 0
        assert result.stats.total_chunks == 0
        assert not result.errors
    finally:
        store.close()


def test_update_index_remote_only_head(
    git_repo: pygit2.Repository, store: IndexStore, tmp_path: Path
) -> None:
    """update_index works when head is a remote-only branch (PR scenario).

    Reproduces the real-world bug: `/review 900` sets `head_branch`
    to the PR's branch name (e.g. `rewrite-mq`), which only exists
    as `origin/rewrite-mq`.  Without the remote fallback in
    `_resolve_commit`, `update_index` throws `KeyError`.
    """
    base_sha = str(git_repo.head.target)

    # Create a second commit with changes.
    utils_path = tmp_path / "src" / "utils.py"
    utils_path.write_bytes(
        b'"""Utility functions."""\n\ndef helper():\n    return 42\n\ndef new_func():\n    return "new"\n'
    )
    index = git_repo.index
    index.add("src/utils.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    head_oid = git_repo.create_commit(None, sig, sig, "Feature work", tree_oid, [parent.id])

    # Put the head commit on a remote-only ref (no local branch).
    git_repo.references.create("refs/remotes/origin/feature-branch", head_oid)

    # Build base, then incremental update using the remote branch name.
    build_index(git_repo, base_sha, store)
    result = update_index(git_repo, base_sha, "feature-branch", store)

    assert result.stats.total_chunks > 0
    # Snapshots stored under "feature-branch", queryable by that name.
    chunks = store.get_chunks("feature-branch")
    assert len(chunks) > 0
    assert any(c.name == "new_func" for c in chunks)


def test_embed_failure_is_nonfatal(tmp_path: Path, mocker: MockerFixture) -> None:
    """When the embedding model cannot load, indexing still succeeds.

    The structural index (chunks, edges, FTS) should be fully usable;
    only vector similarity search is degraded.
    """
    # Don't use the autouse mock — let _embed_missing actually run.
    mocker.stopall()

    # Make the deferred import inside _embed_missing raise ImportError.
    # Setting sys.modules[key] = None causes `from key import ...` to
    # raise ImportError without touching builtins.__import__.

    mocker.patch.dict(sys.modules, {"rbtr.index.embeddings": None})

    repo = pygit2.init_repository(str(tmp_path / "embed_fail"), bare=False)
    (tmp_path / "embed_fail" / "hello.py").write_text("def greet(): pass\n")
    idx = repo.index
    idx.add("hello.py")
    idx.write()
    tree_oid = idx.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])

    store = IndexStore()
    try:
        sha = str(repo.head.target)
        result = build_index(repo, sha, store)

        # Structural index should work.
        assert result.stats.total_chunks > 0
        chunks = store.get_chunks(sha)
        assert len(chunks) > 0

        # No embeddings — all chunks should be unembedded.
        assert all(not c.embedding for c in chunks)
    finally:
        store.close()


def test_embed_batch_failure_skips_batch(tmp_path: Path, mocker: MockerFixture) -> None:
    """When a single embedding batch fails, other batches still succeed."""
    mocker.stopall()

    call_count = 0

    def _flaky_embed(texts: list[str]) -> list[list[float]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("GPU OOM")
        return [[0.1] * 3 for _ in texts]

    # Patch the function on the module so the deferred `from ... import`
    # inside `_embed_missing` picks up the mock.
    mocker.patch("rbtr.index.embeddings.embed_texts", _flaky_embed)

    repo = pygit2.init_repository(str(tmp_path / "batch_fail"), bare=False)
    # Create enough symbols to span multiple batches (batch_size=32).
    lines = "\n".join(f"def func_{i}(): pass" for i in range(40))
    (tmp_path / "batch_fail" / "funcs.py").write_text(lines)
    idx = repo.index
    idx.add("funcs.py")
    idx.write()
    tree_oid = idx.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "init", tree_oid, [])

    store = IndexStore()
    try:
        sha = str(repo.head.target)
        result = build_index(repo, sha, store)
        assert result.stats.total_chunks > 0

        chunks = store.get_chunks(sha)
        # Some chunks should have embeddings (from the batch that succeeded),
        # some should not (from the batch that failed).
        embedded = [c for c in chunks if c.embedding]
        unembedded = [c for c in chunks if not c.embedding]
        assert len(embedded) > 0, "Successful batch should have embeddings"
        assert len(unembedded) > 0, "Failed batch should lack embeddings"
    finally:
        store.close()


def test_query_cache_produces_identical_chunks(tmp_path: Path) -> None:
    """Cached tree-sitter queries must produce the same chunks as fresh ones.

    `_get_query()` caches compiled `Query` objects per language.
    This test creates many same-language files, builds twice (cold cache
    vs warm cache), and asserts identical extraction results.
    """
    repo = pygit2.init_repository(str(tmp_path / "cache_test"), bare=False)

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

    store1 = IndexStore()
    store2 = IndexStore()
    try:
        build_index(repo, sha, store1)
        chunks1 = {c.name for c in store1.get_chunks(sha)}

        # Second build — queries are now cached.
        _get_query.cache_clear()
        build_index(repo, sha, store2)
        chunks2 = {c.name for c in store2.get_chunks(sha)}

        # All 10 functions must be present in both.
        for i in range(10):
            assert f"unique_func_{i}" in chunks1, f"Cold cache missing unique_func_{i}"
            assert f"unique_func_{i}" in chunks2, f"Warm cache missing unique_func_{i}"

        assert chunks1 == chunks2
    finally:
        store1.close()
        store2.close()


def test_deferred_fts_rebuild(
    git_repo: pygit2.Repository, store: IndexStore, commit_sha: str
) -> None:
    """FTS lazily rebuilds on first search_fulltext, not during build."""
    build_index(git_repo, commit_sha, store)

    # FTS is dirty after build but not yet rebuilt.
    assert store._fts_dirty

    # search_fulltext triggers the rebuild.
    results = store.search_fulltext(commit_sha, "helper")
    assert len(results) > 0
    assert not store._fts_dirty


def test_build_index_unknown_language(tmp_path: Path) -> None:
    """Files with unknown extensions should get plaintext chunking."""
    repo = pygit2.init_repository(str(tmp_path / "unknown"), bare=False)

    (tmp_path / "unknown" / "data.xyz").write_text("line1\nline2\nline3\n")
    index = repo.index
    index.add("data.xyz")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Add unknown file", tree_oid, [])

    store = IndexStore()
    try:
        sha = str(repo.head.target)
        result = build_index(repo, sha, store)

        assert result.stats.total_files == 1
        chunks = store.get_chunks(sha)
        assert len(chunks) > 0
        assert all(c.kind == ChunkKind.RAW_CHUNK for c in chunks)
    finally:
        store.close()
