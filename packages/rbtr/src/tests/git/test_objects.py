"""Tests for git object-store operations.

Tests are organised around a shared multi-commit repository
(`sample_repo` from conftest) that has adds, modifications,
deletions, and binary files across three commits.  Individual
tests verify *behaviours* against this dataset rather than
constructing throwaway repos for each assertion.

Functions under test:
- resolve_commit
- walk_tree / list_files
- read_blob
- changed_files
- is_binary / _matches_globs
"""

from __future__ import annotations

from pathlib import Path

import pathspec
import pygit2
import pytest

from rbtr.git import (
    FileEntry,
    _matches_globs,
    changed_files,
    is_binary,
    list_files,
    read_blob,
    read_head,
    resolve_commit,
    walk_tree,
)
from rbtr.rbtrignore import parse_ignore

from .conftest import (
    MergeRepo,
    SampleRepo,
    make_commit,
)


def _lf(
    repo: pygit2.Repository,
    ref: str,
    *,
    max_file_size: int = 1_000_000,
    ignore: pathspec.PathSpec | None = None,
) -> list[FileEntry]:
    """Invokes ``list_files`` and collects the iterator into a list.

    Not setup: ``list_files`` is the system under test.  This is a
    pure wrapper that provides a default ``max_file_size`` and
    materialises the result, expressed once instead of repeating at
    every call site.
    """
    return list(
        list_files(
            repo,
            ref,
            max_file_size=max_file_size,
            ignore=ignore,
        )
    )


# ── _matches_globs ───────────────────────────────────────────────────


def test_matches_globs_hit() -> None:
    assert _matches_globs("style.min.css", ["*.min.css", "*.map"])


def test_matches_globs_miss() -> None:
    assert not _matches_globs("style.css", ["*.min.css", "*.map"])


def test_matches_globs_empty_patterns() -> None:
    assert not _matches_globs("anything.py", [])


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


# ── resolve_commit ───────────────────────────────────────────────────


def test_resolve_by_sha(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, str(sample_repo.base))
    assert commit.id == sample_repo.base


def test_resolve_by_branch_name(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, "feature")
    assert commit.id == sample_repo.head


def test_resolve_falls_back_to_origin(sample_repo: SampleRepo) -> None:
    """When a local branch doesn't exist, origin/<name> is tried."""
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/remote-only", sample_repo.mid)

    commit = resolve_commit(repo, "remote-only")
    assert commit.id == sample_repo.mid


def test_resolve_prefers_local_over_remote(sample_repo: SampleRepo) -> None:
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/main", sample_repo.head)

    # "main" resolves to local (base), not origin (head).
    commit = resolve_commit(repo, "main")
    assert commit.id == sample_repo.base


def test_resolve_unknown_ref_raises(sample_repo: SampleRepo) -> None:
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        resolve_commit(sample_repo.repo, "nonexistent")


# ── read_blob ────────────────────────────────────────────────────────


def test_read_blob_existing_file(sample_repo: SampleRepo, handler_v1: bytes) -> None:
    blob = read_blob(sample_repo.repo, str(sample_repo.base), "src/handler.py")
    assert blob is not None
    assert blob.data == handler_v1


def test_read_blob_nested_path(sample_repo: SampleRepo, utils_content: bytes) -> None:
    blob = read_blob(sample_repo.repo, str(sample_repo.base), "src/utils.py")
    assert blob is not None
    assert blob.data == utils_content


def test_read_blob_missing_file(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, str(sample_repo.base), "nonexistent.py")
    assert blob is None


def test_read_blob_missing_ref(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, "nonexistent-branch", "src/handler.py")
    assert blob is None


def test_read_blob_binary_file(sample_repo: SampleRepo, binary_png_content: bytes) -> None:
    blob = read_blob(sample_repo.repo, "feature", "binary.png")
    assert blob is not None
    assert blob.data == binary_png_content


def test_read_blob_deleted_file(sample_repo: SampleRepo) -> None:
    """readme.md exists at base but is deleted at head."""
    assert read_blob(sample_repo.repo, str(sample_repo.base), "readme.md") is not None
    assert read_blob(sample_repo.repo, "feature", "readme.md") is None


def test_read_blob_by_branch_name(sample_repo: SampleRepo, readme_content: bytes) -> None:
    blob = read_blob(sample_repo.repo, "main", "readme.md")
    assert blob is not None
    assert blob.data == readme_content


# ── list_files ───────────────────────────────────────────────────────


def test_list_files_at_base(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "main")
    paths = {e.path for e in entries}
    assert paths == {"src/handler.py", "src/utils.py", "readme.md"}


def test_list_files_at_head(sample_repo: SampleRepo, config_path: Path) -> None:
    """Head has no readme.md but has config.yaml.  binary.png is skipped."""
    entries = _lf(sample_repo.repo, "feature")
    paths = {e.path for e in entries}
    assert paths == {"src/handler.py", "src/utils.py", "config.yaml"}
    assert "binary.png" not in paths  # binary → skipped
    assert "readme.md" not in paths  # deleted


def test_list_files_returns_file_entries(
    sample_repo: SampleRepo, config_path: Path, handler_v1: bytes
) -> None:
    entries = _lf(sample_repo.repo, "main")
    assert all(isinstance(e, FileEntry) for e in entries)
    handler = next(e for e in entries if e.path == "src/handler.py")
    assert handler.content == handler_v1
    assert handler.blob_sha  # non-empty


def test_list_files_skips_binary(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "feature")
    paths = {e.path for e in entries}
    assert "binary.png" not in paths


def test_list_files_max_file_size(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "main", max_file_size=10)
    # All sample files are > 10 bytes
    assert entries == []


def test_list_files_rbtrignore(sample_repo: SampleRepo, config_path: Path) -> None:
    ignore = parse_ignore("*.yaml\n")
    entries = _lf(sample_repo.repo, "feature", ignore=ignore)
    paths = {e.path for e in entries}
    assert "config.yaml" not in paths


def test_list_files_rbtrignore_negation(sample_repo: SampleRepo, config_path: Path) -> None:
    ignore = parse_ignore("*.yaml\n!config.yaml\n")
    entries = _lf(sample_repo.repo, "feature", ignore=ignore)
    paths = {e.path for e in entries}
    assert "config.yaml" in paths


def test_list_files_gitignore(sample_repo: SampleRepo, config_path: Path) -> None:
    workdir = Path(sample_repo.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / ".gitignore").write_text("config.yaml\n")

    entries = _lf(sample_repo.repo, "feature")
    paths = {e.path for e in entries}
    assert "config.yaml" not in paths


def test_list_files_unresolvable_ref_raises(sample_repo: SampleRepo, config_path: Path) -> None:
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        _lf(sample_repo.repo, "nonexistent-branch")


def test_list_files_remote_branch(sample_repo: SampleRepo, config_path: Path) -> None:
    """Resolves origin/<name> when local branch doesn't exist."""
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/remote-only", sample_repo.mid)

    entries = _lf(repo, "remote-only")
    paths = {e.path for e in entries}
    assert "config.yaml" in paths  # added at mid


# ── changed_files ────────────────────────────────────────────────────


def test_changed_files_base_to_head(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo, "main", "feature")
    # handler.py (modified), readme.md (deleted), config.yaml (added), binary.png (added)
    assert paths == {"src/handler.py", "readme.md", "config.yaml", "binary.png"}


def test_changed_files_base_to_mid(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo, str(sample_repo.base), str(sample_repo.mid))
    assert paths == {"src/handler.py", "config.yaml"}


def test_changed_files_identical(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo, "main", "main")
    assert paths == set()


def test_changed_files_with_remote_branch(sample_repo: SampleRepo) -> None:
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/pr-head", sample_repo.head)
    paths = changed_files(repo, "main", "pr-head")
    assert "src/handler.py" in paths
    assert "readme.md" in paths


def test_merge_changed_files_only_pr_changes(merge_repo: MergeRepo) -> None:
    """changed_files base..head returns only `feature.py`."""
    paths = changed_files(merge_repo.repo, "base", "head")
    assert paths == {"feature.py"}


# ── walk_tree ────────────────────────────────────────────────────────


def test_walk_tree_yields_all_blobs(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, "main")
    pairs = list(walk_tree(sample_repo.repo, commit.tree, ""))
    paths = {p for p, _ in pairs}
    assert paths == {"src/handler.py", "src/utils.py", "readme.md"}


def test_walk_tree_blobs_are_readable(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, "main")
    for _path, blob in walk_tree(sample_repo.repo, commit.tree, ""):
        assert isinstance(blob, pygit2.Blob)
        assert len(blob.data) > 0


# ── Edge cases with standalone repos ─────────────────────────────────


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    """A bare repo for one-off edge case tests."""
    return pygit2.init_repository(str(tmp_path / "edge"))


def test_list_files_rbtrignore_still_skips_binary(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    """Binary files are skipped even without an ignore spec."""
    oid = make_commit(git_repo, {"image.png": b"\x89PNG\r\n\x1a\n\x00\x00"})
    paths = {e.path for e in _lf(git_repo, str(oid))}
    assert paths == set()


def test_list_files_rbtrignore_still_skips_oversized(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    """Oversized files are skipped even without an ignore spec."""
    oid = make_commit(git_repo, {"big.py": b"x" * 200})
    paths = {e.path for e in _lf(git_repo, str(oid), max_file_size=50)}
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
    pygit2.init_repository(str(path))
    assert read_head(str(path)) is None


def test_read_head_missing_path_returns_none(tmp_path: Path) -> None:
    """Non-git paths yield None, not an exception."""
    assert read_head(str(tmp_path / "does-not-exist")) is None
