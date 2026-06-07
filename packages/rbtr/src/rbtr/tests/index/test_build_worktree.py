"""Tests for worktree indexing via `build_index(repo_path, tree_sha, ...)`."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.git import changed_files, worktree_tree_sha
from rbtr.index.models import RepoRef
from rbtr.index.orchestrator import build_index
from rbtr.index.store import IndexStore


@pytest.fixture
def worktree_repo(git_repo: pygit2.Repository, tmp_path: Path) -> tuple[pygit2.Repository, str]:
    """The shared `git_repo` with dirty working-tree changes.

    Modifications applied on top of the initial commit:
    - `src/utils.py` modified: `helper()` returns 99 instead of 42
    - `src/service.py` added: new file with `serve()`
    - `tests/test_utils.py` deleted from disk

    Returns `(repo, head_sha)`.
    """
    head_sha = str(git_repo.head.target)

    # Modify existing file.
    (tmp_path / "src" / "utils.py").write_bytes(b"""\
\"\"\"Utility functions.\"\"\"

def helper():
    return 99

def format_name(name):
    return name.strip()
""")

    # Add new file.
    (tmp_path / "src" / "service.py").write_bytes(b"""\
\"\"\"Service layer.\"\"\"

def serve():
    return True
""")

    # Delete existing file.
    (tmp_path / "tests" / "test_utils.py").unlink()

    return git_repo, head_sha


@pytest.fixture
def wt_sha(worktree_repo: tuple[pygit2.Repository, str]) -> str:
    """The worktree tree SHA for the dirty `worktree_repo`."""
    repo, _ = worktree_repo
    sha = worktree_tree_sha(repo.workdir)
    # worktree_repo always has dirty files.
    assert sha is not None
    return sha


# ── Build tests ──────────────────────────────────────────────────────


def test_build_worktree_creates_chunks(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    repo, _ = worktree_repo
    result = build_index(repo.workdir, wt_sha, store, repo_id=1)
    assert result.stats.total_chunks > 0


def test_build_worktree_blob_dedup(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """Unchanged files reuse HEAD blobs — no duplicate extraction."""
    repo, head_sha = worktree_repo

    # Build HEAD first, then worktree.
    r1 = build_index(repo.workdir, head_sha, store, repo_id=1)
    wt = build_index(repo.workdir, wt_sha, store, repo_id=1)

    # Files unchanged between HEAD and worktree (e.g. models.py,
    # main.py, README.md) share blobs, so fewer files are parsed
    # than the full-build baseline.
    assert wt.stats.parsed_files < r1.stats.parsed_files


def test_build_worktree_modified_visible(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """Modified file content appears in worktree chunks."""
    repo, _ = worktree_repo
    build_index(repo.workdir, wt_sha, store, repo_id=1)

    chunks = store.get_chunks(wt_sha, repo_id=1)
    helper_chunks = [c for c in chunks if c.name == "helper"]
    assert len(helper_chunks) == 1
    assert "99" in helper_chunks[0].content


def test_build_worktree_added_file(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """New working-tree file appears in worktree index."""
    repo, _ = worktree_repo
    build_index(repo.workdir, wt_sha, store, repo_id=1)

    chunks = store.get_chunks(wt_sha, repo_id=1)
    service_chunks = [c for c in chunks if c.file_path == "src/service.py"]
    assert len(service_chunks) > 0


def test_build_worktree_deleted_file(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """Deleted working-tree file is absent from worktree index."""
    repo, _ = worktree_repo
    build_index(repo.workdir, wt_sha, store, repo_id=1)

    chunks = store.get_chunks(wt_sha, repo_id=1)
    test_chunks = [c for c in chunks if c.file_path == "tests/test_utils.py"]
    assert test_chunks == []


def test_build_worktree_edges(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """Worktree build infers edges (import relationships)."""
    repo, _ = worktree_repo
    result = build_index(repo.workdir, wt_sha, store, repo_id=1)
    assert result.stats.total_edges > 0


def test_build_worktree_marks_indexed(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """Tree SHA appears in `list_indexed_commits` after build."""
    repo, _ = worktree_repo
    build_index(repo.workdir, wt_sha, store, repo_id=1)

    indexed = [sha for sha, _ts in store.list_indexed_commits(repo_id=1)]
    assert wt_sha in indexed


# ── Integration tests ────────────────────────────────────────────────


def test_search_returns_worktree_content(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """FTS search against tree SHA finds the modified content."""
    repo, _ = worktree_repo
    build_index(repo.workdir, wt_sha, store, repo_id=1)

    results = store.search([RepoRef(repo_id=1, commit_sha=wt_sha)], "helper")
    helpers = [r for r in results if r.name == "helper"]
    assert len(helpers) == 1
    assert "99" in helpers[0].content


def test_head_still_visible_after_worktree_build(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str
) -> None:
    """HEAD index is not clobbered by the worktree build."""
    repo, head_sha = worktree_repo

    build_index(repo.workdir, head_sha, store, repo_id=1)
    build_index(repo.workdir, wt_sha, store, repo_id=1)

    # HEAD still has the original helper() returning 42.
    head_chunks = store.get_chunks(head_sha, repo_id=1)
    helpers = [c for c in head_chunks if c.name == "helper"]
    assert len(helpers) == 1
    assert "42" in helpers[0].content

    # HEAD still has the deleted test file.
    test_chunks = [c for c in head_chunks if c.file_path == "tests/test_utils.py"]
    assert len(test_chunks) > 0


# ── Unhappy paths / edge cases ───────────────────────────────────────


def test_build_worktree_clean_matches_head(git_repo: pygit2.Repository, store: IndexStore) -> None:
    """Clean worktree produces the same chunks as HEAD when built with HEAD's tree SHA."""
    head_sha = str(git_repo.head.target)
    # Clean tree: worktree_tree_sha returns None, so we use HEAD's tree directly.
    head_tree_sha = str(git_repo.head.peel(pygit2.Tree).id)

    build_index(git_repo.workdir, head_sha, store, repo_id=1)
    build_index(git_repo.workdir, head_tree_sha, store, repo_id=1)

    head_chunks = store.get_chunks(head_sha, repo_id=1)
    wt_chunks = store.get_chunks(head_tree_sha, repo_id=1)

    head_names = sorted(c.name for c in head_chunks)
    wt_names = sorted(c.name for c in wt_chunks)
    assert head_names == wt_names

    head_content = sorted(c.content for c in head_chunks)
    wt_content = sorted(c.content for c in wt_chunks)
    assert head_content == wt_content


def test_rebuild_worktree_reflects_new_edits(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str, tmp_path: Path
) -> None:
    """Rebuilding the worktree index picks up subsequent edits."""
    repo, _ = worktree_repo

    # First build: helper() returns 99.
    build_index(repo.workdir, wt_sha, store, repo_id=1)
    chunks_v1 = store.get_chunks(wt_sha, repo_id=1)
    helpers_v1 = [c for c in chunks_v1 if c.name == "helper"]
    assert "99" in helpers_v1[0].content

    # Edit again: helper() returns 200.
    (tmp_path / "src" / "utils.py").write_bytes(b"""\
\"\"\"Utility functions.\"\"\"

def helper():
    return 200

def format_name(name):
    return name.strip()
""")

    # New tree SHA — different from v1.
    tree_sha_v2 = worktree_tree_sha(repo.workdir)
    assert tree_sha_v2 is not None
    assert wt_sha != tree_sha_v2

    # Rebuild picks up the new content.
    build_index(repo.workdir, tree_sha_v2, store, repo_id=1)
    chunks_v2 = store.get_chunks(tree_sha_v2, repo_id=1)
    helpers_v2 = [c for c in chunks_v2 if c.name == "helper"]
    assert len(helpers_v2) == 1
    assert "200" in helpers_v2[0].content
    assert "99" not in helpers_v2[0].content


def test_build_worktree_file_unreadable_mid_walk(
    worktree_repo: tuple[pygit2.Repository, str], store: IndexStore, wt_sha: str, tmp_path: Path
) -> None:
    """A dirty file that becomes unreadable after tree SHA computation.

    The tree SHA was computed when the file was readable, but by the
    time list_files walks the tree, the blob is in the object store
    (written by add_all). So the file IS visible in the tree — the
    blob was already captured. This is different from the old
    _list_worktree_files path which read from disk.
    """
    repo, _ = worktree_repo

    # Now break the file — but the blob is already in git's object store.
    utils = tmp_path / "src" / "utils.py"
    utils.unlink()
    utils.symlink_to("/nonexistent/target")

    # Build should complete — the tree object has the blob.
    result = build_index(repo.workdir, wt_sha, store, repo_id=1)
    assert result.stats.total_chunks > 0

    # utils.py IS present because its blob was written to the object
    # store by add_all() before we broke the symlink.
    chunks = store.get_chunks(wt_sha, repo_id=1)
    utils_chunks = [c for c in chunks if c.file_path == "src/utils.py"]
    assert len(utils_chunks) > 0


def test_changed_files_same_tree_sha(
    git_repo: pygit2.Repository,
) -> None:
    """Diffing a tree SHA against itself returns no changes."""
    # Use HEAD's tree SHA as a stand-in (clean tree).
    head_tree_sha = str(git_repo.head.peel(pygit2.Tree).id)
    result = changed_files(git_repo.workdir, head_tree_sha, head_tree_sha)
    assert result == set()
