"""Tests for git object-store operations.

Tests are organised around a shared multi-commit repository
(`sample_repo` from conftest) that has adds, modifications,
deletions, and binary files across three commits.  Individual
tests verify *behaviours* against this dataset rather than
constructing throwaway repos for each assertion.

Functions under test:
- list_files (including tree SHA path)
- read_blob
- changed_files
- worktree_tree_sha
- is_binary
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.errors import RbtrError
from rbtr.git import (
    FileEntry,
    changed_files,
    is_binary,
    list_files,
    read_blob,
    read_head,
    worktree_tree_sha,
)
from rbtr.rbtrignore import parse_ignore

from ..conftest import make_commit
from .conftest import (
    MergeRepo,
    SampleRepo,
)

# ── is_binary ───────────────────────────────────────────────────────


def test_is_binary_with_null_byte() -> None:
    assert is_binary(b"hello\x00world")


def test_is_binary_text() -> None:
    assert not is_binary(b"hello world\n")


def test_is_binary_empty() -> None:
    assert not is_binary(b"")


def test_is_binary_null_beyond_sample() -> None:
    data = b"a" * 100 + b"\x00"
    assert not is_binary(data, sample_size=50)


# ── read_blob ────────────────────────────────────────────────────────


def test_read_blob_existing_file(sample_repo: SampleRepo, handler_v1: bytes) -> None:
    blob = read_blob(sample_repo.repo.workdir, str(sample_repo.base), "src/handler.py")
    assert blob is not None
    assert blob.data == handler_v1


def test_read_blob_nested_path(sample_repo: SampleRepo, utils_content: bytes) -> None:
    blob = read_blob(sample_repo.repo.workdir, str(sample_repo.base), "src/utils.py")
    assert blob is not None
    assert blob.data == utils_content


def test_read_blob_missing_file(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo.workdir, str(sample_repo.base), "nonexistent.py")
    assert blob is None


def test_read_blob_missing_ref(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo.workdir, "nonexistent-branch", "src/handler.py")
    assert blob is None


def test_read_blob_binary_file(sample_repo: SampleRepo, binary_png_content: bytes) -> None:
    blob = read_blob(sample_repo.repo.workdir, "feature", "binary.png")
    assert blob is not None
    assert blob.data == binary_png_content


def test_read_blob_deleted_file(sample_repo: SampleRepo) -> None:
    """readme.md exists at base but is deleted at head."""
    assert read_blob(sample_repo.repo.workdir, str(sample_repo.base), "readme.md") is not None
    assert read_blob(sample_repo.repo.workdir, "feature", "readme.md") is None


def test_read_blob_by_branch_name(sample_repo: SampleRepo, readme_content: bytes) -> None:
    blob = read_blob(sample_repo.repo.workdir, "main", "readme.md")
    assert blob is not None
    assert blob.data == readme_content


# ── list_files ───────────────────────────────────────────────────────


def test_list_files_at_base(sample_repo: SampleRepo) -> None:
    entries = list(list_files(sample_repo.repo.workdir, "main", max_file_size=1_000_000))
    paths = {e.path for e in entries}
    assert paths == {"src/handler.py", "src/utils.py", "readme.md"}


def test_list_files_at_head(sample_repo: SampleRepo) -> None:
    """Head has no readme.md but has config.yaml.  binary.png is skipped."""
    entries = list(list_files(sample_repo.repo.workdir, "feature", max_file_size=1_000_000))
    paths = {e.path for e in entries}
    assert paths == {"src/handler.py", "src/utils.py", "config.yaml"}
    assert "binary.png" not in paths  # binary → skipped
    assert "readme.md" not in paths  # deleted


def test_list_files_returns_file_entries(sample_repo: SampleRepo, handler_v1: bytes) -> None:
    entries = list(list_files(sample_repo.repo.workdir, "main", max_file_size=1_000_000))
    assert all(isinstance(e, FileEntry) for e in entries)
    handler = next(e for e in entries if e.path == "src/handler.py")
    assert handler.content == handler_v1
    assert handler.blob_sha  # non-empty


def test_list_files_skips_binary(sample_repo: SampleRepo) -> None:
    entries = list(list_files(sample_repo.repo.workdir, "feature", max_file_size=1_000_000))
    paths = {e.path for e in entries}
    assert "binary.png" not in paths


def test_list_files_max_file_size(sample_repo: SampleRepo) -> None:
    entries = list(list_files(sample_repo.repo.workdir, "main", max_file_size=10))
    # All sample files are > 10 bytes
    assert entries == []


def test_list_files_rbtrignore(sample_repo: SampleRepo) -> None:
    ignore = parse_ignore("*.yaml\n")
    entries = list(
        list_files(sample_repo.repo.workdir, "feature", max_file_size=1_000_000, ignore=ignore)
    )
    paths = {e.path for e in entries}
    assert "config.yaml" not in paths


def test_list_files_rbtrignore_negation(sample_repo: SampleRepo) -> None:
    ignore = parse_ignore("*.yaml\n!config.yaml\n")
    entries = list(
        list_files(sample_repo.repo.workdir, "feature", max_file_size=1_000_000, ignore=ignore)
    )
    paths = {e.path for e in entries}
    assert "config.yaml" in paths


def test_list_files_gitignore(sample_repo: SampleRepo) -> None:
    workdir = Path(sample_repo.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / ".gitignore").write_text("config.yaml\n")

    entries = list(list_files(sample_repo.repo.workdir, "feature", max_file_size=1_000_000))
    paths = {e.path for e in entries}
    assert "config.yaml" not in paths


def test_list_files_unresolvable_ref_raises(sample_repo: SampleRepo) -> None:
    with pytest.raises(RbtrError, match="Cannot resolve ref"):
        list(list_files(sample_repo.repo.workdir, "nonexistent-branch", max_file_size=1_000_000))


def test_list_files_remote_branch(sample_repo: SampleRepo) -> None:
    """Resolves origin/<name> when local branch doesn't exist."""
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/remote-only", sample_repo.mid)

    entries = list(list_files(repo.workdir, "remote-only", max_file_size=1_000_000))
    paths = {e.path for e in entries}
    assert "config.yaml" in paths  # added at mid


# ── changed_files ────────────────────────────────────────────────────


def test_changed_files_base_to_head(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo.workdir, "main", "feature")
    # handler.py (modified), readme.md (deleted), config.yaml (added), binary.png (added)
    assert paths == {"src/handler.py", "readme.md", "config.yaml", "binary.png"}


def test_changed_files_base_to_mid(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo.workdir, str(sample_repo.base), str(sample_repo.mid))
    assert paths == {"src/handler.py", "config.yaml"}


def test_changed_files_identical(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo.workdir, "main", "main")
    assert paths == set()


def test_changed_files_with_remote_branch(sample_repo: SampleRepo) -> None:
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/pr-head", sample_repo.head)
    paths = changed_files(repo.workdir, "main", "pr-head")
    assert "src/handler.py" in paths
    assert "readme.md" in paths


def test_merge_changed_files_only_pr_changes(merge_repo: MergeRepo) -> None:
    """changed_files base..head returns only `feature.py`."""
    paths = changed_files(merge_repo.repo.workdir, "base", "head")
    assert paths == {"feature.py"}


# ── Edge cases with standalone repos ─────────────────────────────────


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    """A bare repo for one-off edge case tests."""
    return pygit2.init_repository(str(tmp_path / "edge"), initial_head="main")


def test_list_files_rbtrignore_still_skips_binary(git_repo: pygit2.Repository) -> None:
    """Binary files are skipped even without an ignore spec."""
    oid = make_commit(git_repo, {"image.png": b"\x89PNG\r\n\x1a\n\x00\x00"})
    paths = {e.path for e in list_files(git_repo.workdir, str(oid), max_file_size=1_000_000)}
    assert paths == set()


def test_list_files_rbtrignore_still_skips_oversized(git_repo: pygit2.Repository) -> None:
    """Oversized files are skipped even without an ignore spec."""
    oid = make_commit(git_repo, {"big.py": b"x" * 200})
    paths = {e.path for e in list_files(git_repo.workdir, str(oid), max_file_size=50)}
    assert paths == set()


def test_read_head_returns_sha_of_current_head(sample_repo: SampleRepo) -> None:
    workdir = sample_repo.repo.workdir
    assert workdir is not None
    sha = read_head(workdir)
    assert sha == str(sample_repo.base)  # main is kept at base


def test_read_head_unborn_returns_none(tmp_path: Path) -> None:
    """A freshly-init'd repo with no commits has no HEAD."""
    path = tmp_path / "empty"
    path.mkdir()
    pygit2.init_repository(str(path), initial_head="main")
    assert read_head(str(path)) is None


def test_read_head_missing_path_returns_none(tmp_path: Path) -> None:
    """Non-git paths yield None, not an exception."""
    assert read_head(str(tmp_path / "does-not-exist")) is None


# ── worktree fixture (shared by worktree_tree_sha and tree SHA tests) ──


@pytest.fixture
def worktree_repo(tmp_path: Path) -> pygit2.Repository:
    """A non-bare repo with files on disk for worktree tests.

    Commits two tracked files (`a.py`, `b.py`) so that
    tree SHA tests have a HEAD tree to compare against.
    """
    repo = pygit2.init_repository(str(tmp_path / "wt"), bare=False, initial_head="main")
    workdir = Path(repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs

    (workdir / "a.py").write_text("def a(): pass\n")
    (workdir / "b.py").write_text("def b(): pass\n")

    idx = repo.index
    idx.add("a.py")
    idx.add("b.py")
    idx.write()
    tree = idx.write_tree()
    sig = pygit2.Signature("T", "t@t")
    repo.create_commit("HEAD", sig, sig, "init", tree, [])

    return repo


# ── worktree_tree_sha ──────────────────────────────────────────────────────


def test_worktree_tree_sha_clean_returns_none(worktree_repo: pygit2.Repository) -> None:
    """Clean tree returns None — tree SHA equals HEAD's tree."""
    assert worktree_tree_sha(worktree_repo.workdir) is None


def test_worktree_tree_sha_dirty_returns_hex(worktree_repo: pygit2.Repository) -> None:
    """Dirty tree returns a 40-char hex SHA."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("def a_v2(): pass\n")

    sha = worktree_tree_sha(worktree_repo.workdir)
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_worktree_tree_sha_deterministic(worktree_repo: pygit2.Repository) -> None:
    """Same working-tree content produces the same SHA."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("def a_v2(): pass\n")

    sha1 = worktree_tree_sha(worktree_repo.workdir)
    sha2 = worktree_tree_sha(worktree_repo.workdir)
    assert sha1 == sha2


def test_worktree_tree_sha_changes_on_edit(worktree_repo: pygit2.Repository) -> None:
    """Editing a file changes the tree SHA."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("v1\n")
    sha1 = worktree_tree_sha(worktree_repo.workdir)

    (workdir / "a.py").write_text("v2\n")
    sha2 = worktree_tree_sha(worktree_repo.workdir)

    assert sha1 is not None
    assert sha2 is not None
    assert sha1 != sha2


def test_worktree_tree_sha_no_staging_side_effect(worktree_repo: pygit2.Repository) -> None:
    """On-disk staging area is not modified by worktree_tree_sha."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("edited\n")

    # Read the index state before.
    worktree_repo.index.read()
    before = {e.path for e in worktree_repo.index}

    worktree_tree_sha(worktree_repo.workdir)

    # Re-read and compare.
    worktree_repo.index.read()
    after = {e.path for e in worktree_repo.index}
    assert before == after


def test_worktree_tree_sha_missing_repo(tmp_path: Path) -> None:
    """Non-git path returns None."""
    assert worktree_tree_sha(str(tmp_path / "no-such-repo")) is None


# ── list_files with tree SHAs ───────────────────────────────────────────


def test_list_files_with_tree_sha(worktree_repo: pygit2.Repository) -> None:
    """list_files walks a tree SHA the same as a commit ref."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("def a_v2(): pass\n")

    tree_sha = worktree_tree_sha(worktree_repo.workdir)
    assert tree_sha is not None

    entries = list(list_files(worktree_repo.workdir, tree_sha, max_file_size=1_000_000))
    paths = {e.path for e in entries}
    assert "a.py" in paths
    assert "b.py" in paths

    # The modified file should have updated content.
    a_entry = next(e for e in entries if e.path == "a.py")
    assert b"a_v2" in a_entry.content


def test_list_files_tree_sha_includes_new_files(worktree_repo: pygit2.Repository) -> None:
    """Tree SHA includes untracked files added via add_all."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "c.py").write_text("def c(): pass\n")

    tree_sha = worktree_tree_sha(worktree_repo.workdir)
    assert tree_sha is not None

    paths = {e.path for e in list_files(worktree_repo.workdir, tree_sha, max_file_size=1_000_000)}
    assert "c.py" in paths


def test_list_files_tree_sha_excludes_deleted(worktree_repo: pygit2.Repository) -> None:
    """Tree SHA excludes deleted files."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "b.py").unlink()

    tree_sha = worktree_tree_sha(worktree_repo.workdir)
    assert tree_sha is not None

    paths = {e.path for e in list_files(worktree_repo.workdir, tree_sha, max_file_size=1_000_000)}
    assert "b.py" not in paths
    assert "a.py" in paths


# ── changed_files with tree SHAs ────────────────────────────────────────


def test_changed_files_commit_vs_tree_sha(worktree_repo: pygit2.Repository) -> None:
    """Diff between a commit and a worktree tree SHA."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("edited\n")

    tree_sha = worktree_tree_sha(worktree_repo.workdir)
    assert tree_sha is not None

    paths = changed_files(worktree_repo.workdir, "HEAD", tree_sha)
    assert "a.py" in paths
    assert "b.py" not in paths


def test_changed_files_same_tree_sha(worktree_repo: pygit2.Repository) -> None:
    """Same tree SHA on both sides returns empty set."""
    workdir = Path(worktree_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / "a.py").write_text("edited\n")

    tree_sha = worktree_tree_sha(worktree_repo.workdir)
    assert tree_sha is not None

    paths = changed_files(worktree_repo.workdir, tree_sha, tree_sha)
    assert paths == set()
