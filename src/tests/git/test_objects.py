"""Tests for git file listing and change detection."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr.config import config
from rbtr.git import FileEntry, changed_files, is_binary, list_files
from rbtr.git.filters import _matches_globs

# ── Helpers ──────────────────────────────────────────────────────────


def _build_tree(
    repo: pygit2.Repository,
    files: dict[str, bytes],
) -> pygit2.Oid:
    """Build a nested tree from ``{"dir/file.py": b"..."}`` paths."""
    # Group files by top-level directory component.
    subtrees: dict[str, dict[str, bytes]] = {}
    blobs: dict[str, bytes] = {}

    for path, content in files.items():
        if "/" in path:
            top, rest = path.split("/", 1)
            subtrees.setdefault(top, {})[rest] = content
        else:
            blobs[path] = content

    tb = repo.TreeBuilder()
    for name, data in blobs.items():
        tb.insert(name, repo.create_blob(data), pygit2.GIT_FILEMODE_BLOB)
    for name, sub_files in subtrees.items():
        tb.insert(name, _build_tree(repo, sub_files), pygit2.GIT_FILEMODE_TREE)
    return tb.write()


def _commit_files(
    repo: pygit2.Repository,
    files: dict[str, bytes],
    *,
    message: str = "commit",
    parents: list[pygit2.Oid] | None = None,
) -> pygit2.Oid:
    """Create a commit with the given file tree and return its OID."""
    tree_oid = _build_tree(repo, files)

    sig = pygit2.Signature("test", "test@test.com")
    return repo.create_commit(
        "refs/heads/main",
        sig,
        sig,
        message,
        tree_oid,
        parents or [],
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    """Create a bare-minimum git repo with one commit."""
    return pygit2.init_repository(str(tmp_path / "repo"))


# ── _matches_globs ───────────────────────────────────────────────────


def test_matches_globs_hit() -> None:
    assert _matches_globs("style.min.css", ["*.min.css", "*.map"])


def test_matches_globs_miss() -> None:
    assert not _matches_globs("style.css", ["*.min.css", "*.map"])


def test_matches_globs_empty_patterns() -> None:
    assert not _matches_globs("anything.py", [])


# ── is_binary ───────────────────────────────────────────────────────


def testis_binary_with_null_byte() -> None:
    assert is_binary(b"hello\x00world")


def testis_binary_text() -> None:
    assert not is_binary(b"hello world\n")


def testis_binary_empty() -> None:
    assert not is_binary(b"")


def testis_binary_null_beyond_sample() -> None:
    data = b"a" * 100 + b"\x00"
    assert not is_binary(data, sample_size=50)


# ── list_files basic ─────────────────────────────────────────────────


def test_list_files_returns_text_files(git_repo: pygit2.Repository, config_path: Path) -> None:
    oid = _commit_files(
        git_repo,
        {
            "hello.py": b"print('hi')\n",
            "readme.md": b"# Hello\n",
        },
    )
    entries = list(list_files(git_repo, str(oid)))
    paths = {e.path for e in entries}
    assert paths == {"hello.py", "readme.md"}
    assert all(isinstance(e, FileEntry) for e in entries)


def test_list_files_skips_binary(git_repo: pygit2.Repository, config_path: Path) -> None:
    oid = _commit_files(
        git_repo,
        {
            "image.png": b"\x89PNG\r\n\x1a\n\x00\x00",
            "code.py": b"x = 1\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == {"code.py"}


def test_list_files_skips_large_files(git_repo: pygit2.Repository, config_path: Path) -> None:
    config.index.max_file_size = 100
    oid = _commit_files(
        git_repo,
        {
            "big.py": b"x" * 200,
            "small.py": b"x = 1\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == {"small.py"}


def test_list_files_max_file_size_override(git_repo: pygit2.Repository, config_path: Path) -> None:
    config.index.max_file_size = 1000
    oid = _commit_files(
        git_repo,
        {
            "big.py": b"x" * 200,
        },
    )
    # Override with a smaller limit via kwarg.
    paths = {e.path for e in list_files(git_repo, str(oid), max_file_size=50)}
    assert paths == set()


# ── extend_exclude ───────────────────────────────────────────────────


def test_list_files_extend_exclude(git_repo: pygit2.Repository, config_path: Path) -> None:
    config.index.extend_exclude = ["*.lock"]
    oid = _commit_files(
        git_repo,
        {
            "app.py": b"x = 1\n",
            "poetry.lock": b"lots of lock data\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == {"app.py"}


def test_list_files_extend_exclude_empty(git_repo: pygit2.Repository, config_path: Path) -> None:
    config.index.extend_exclude = []
    oid = _commit_files(
        git_repo,
        {
            "app.py": b"x = 1\n",
            "style.min.css": b"body{}\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert "style.min.css" in paths


# ── gitignore ────────────────────────────────────────────────────────


def test_list_files_respects_gitignore(git_repo: pygit2.Repository, config_path: Path) -> None:
    config.index.extend_exclude = []
    # Write a .gitignore to the working directory so pygit2 sees it.
    workdir = Path(git_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs return Optional
    (workdir / ".gitignore").write_text("ignored.py\n")

    oid = _commit_files(
        git_repo,
        {
            "kept.py": b"x = 1\n",
            "ignored.py": b"y = 2\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert "kept.py" in paths
    assert "ignored.py" not in paths


# ── include (force-include overrides gitignore) ──────────────────────


def test_list_files_include_overrides_gitignore(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    config.index.extend_exclude = []
    config.index.include = ["ignored.py"]
    workdir = Path(git_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs return Optional
    (workdir / ".gitignore").write_text("ignored.py\n")

    oid = _commit_files(
        git_repo,
        {
            "kept.py": b"x = 1\n",
            "ignored.py": b"y = 2\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == {"kept.py", "ignored.py"}


def test_list_files_include_overrides_extend_exclude(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    config.index.extend_exclude = ["*.lock"]
    config.index.include = ["important.lock"]
    oid = _commit_files(
        git_repo,
        {
            "app.py": b"x = 1\n",
            "important.lock": b"keep me\n",
            "other.lock": b"skip me\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == {"app.py", "important.lock"}


def test_list_files_include_still_skips_binary(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    config.index.include = ["image.png"]
    oid = _commit_files(
        git_repo,
        {
            "image.png": b"\x89PNG\r\n\x1a\n\x00\x00",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == set()


def test_list_files_include_still_skips_oversized(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    config.index.max_file_size = 50
    config.index.include = ["big.py"]
    oid = _commit_files(
        git_repo,
        {
            "big.py": b"x" * 200,
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == set()


# ── changed_files ────────────────────────────────────────────────────


def test_changed_files_detects_added(git_repo: pygit2.Repository, config_path: Path) -> None:
    base = _commit_files(git_repo, {"a.py": b"x = 1\n"})
    head = _commit_files(
        git_repo,
        {"a.py": b"x = 1\n", "b.py": b"y = 2\n"},
        parents=[base],
    )
    paths = changed_files(git_repo, str(base), str(head))
    assert "b.py" in paths


def test_changed_files_detects_modified(git_repo: pygit2.Repository, config_path: Path) -> None:
    base = _commit_files(git_repo, {"a.py": b"x = 1\n"})
    head = _commit_files(
        git_repo,
        {"a.py": b"x = 2\n"},
        parents=[base],
    )
    paths = changed_files(git_repo, str(base), str(head))
    assert "a.py" in paths


def test_changed_files_detects_deleted(git_repo: pygit2.Repository, config_path: Path) -> None:
    base = _commit_files(git_repo, {"a.py": b"x = 1\n", "b.py": b"y = 2\n"})
    head = _commit_files(
        git_repo,
        {"a.py": b"x = 1\n"},
        parents=[base],
    )
    paths = changed_files(git_repo, str(base), str(head))
    assert "b.py" in paths


def test_changed_files_empty_when_identical(git_repo: pygit2.Repository, config_path: Path) -> None:
    base = _commit_files(git_repo, {"a.py": b"x = 1\n"})
    head = _commit_files(
        git_repo,
        {"a.py": b"x = 1\n"},
        parents=[base],
    )
    paths = changed_files(git_repo, str(base), str(head))
    assert paths == set()


# ── Nested directories ───────────────────────────────────────────────


def test_list_files_nested_dirs(git_repo: pygit2.Repository, config_path: Path) -> None:
    oid = _commit_files(
        git_repo,
        {
            "src/app.py": b"x = 1\n",
            "src/lib/util.py": b"y = 2\n",
            "readme.md": b"# Hi\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == {"src/app.py", "src/lib/util.py", "readme.md"}


def test_list_files_gitignore_nested_dir(git_repo: pygit2.Repository, config_path: Path) -> None:
    config.index.extend_exclude = []
    workdir = Path(git_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs return Optional
    (workdir / ".gitignore").write_text("vendor/\n")

    oid = _commit_files(
        git_repo,
        {
            "app.py": b"x = 1\n",
            "vendor/lib.py": b"y = 2\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert "app.py" in paths
    assert "vendor/lib.py" not in paths


def test_list_files_include_overrides_gitignore_nested(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    config.index.extend_exclude = []
    config.index.include = ["vendor/important.py"]
    workdir = Path(git_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs return Optional
    (workdir / ".gitignore").write_text("vendor/\n")

    oid = _commit_files(
        git_repo,
        {
            "app.py": b"x = 1\n",
            "vendor/important.py": b"keep\n",
            "vendor/junk.py": b"skip\n",
        },
    )
    paths = {e.path for e in list_files(git_repo, str(oid))}
    assert paths == {"app.py", "vendor/important.py"}


# ── FileEntry fields ─────────────────────────────────────────────────


def test_list_files_entry_content(git_repo: pygit2.Repository, config_path: Path) -> None:
    oid = _commit_files(git_repo, {"hello.py": b"print('hi')\n"})
    entries = list(list_files(git_repo, str(oid)))
    assert len(entries) == 1
    assert entries[0].path == "hello.py"
    assert entries[0].content == b"print('hi')\n"
    assert entries[0].blob_sha  # non-empty string


# ── Remote branch fallback ───────────────────────────────────────────


def test_list_files_resolves_remote_branch(git_repo: pygit2.Repository, config_path: Path) -> None:
    """list_files resolves ``origin/<branch>`` when the local branch
    doesn't exist — common for PR head branches.
    """
    oid = _commit_files(git_repo, {"app.py": b"x = 1\n"})

    # Create a remote ref that mimics ``git fetch origin feature``.
    git_repo.references.create("refs/remotes/origin/feature", oid)

    # "feature" doesn't exist as a local branch, but origin/feature does.
    entries = list(list_files(git_repo, "feature"))
    assert {e.path for e in entries} == {"app.py"}


def test_list_files_unresolvable_ref_raises(git_repo: pygit2.Repository, config_path: Path) -> None:
    _commit_files(git_repo, {"app.py": b"x = 1\n"})
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        list(list_files(git_repo, "nonexistent-branch"))


def test_changed_files_with_remote_branch(git_repo: pygit2.Repository, config_path: Path) -> None:
    """changed_files works when the head ref is a remote-only branch.

    This is the exact scenario for PR reviews: base is a local branch
    (``main``), head is remote-only (``origin/feature``).
    """
    base = _commit_files(git_repo, {"a.py": b"x = 1\n"})
    head = _commit_files(
        git_repo,
        {"a.py": b"x = 1\n", "b.py": b"y = 2\n"},
        parents=[base],
    )

    # Move main back to base, put head on a remote-only ref.
    git_repo.references.create("refs/heads/main", base, force=True)
    git_repo.references.create("refs/remotes/origin/feature", head)

    paths = changed_files(git_repo, "main", "feature")
    assert "b.py" in paths


def test_list_files_prefers_local_over_remote(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    """When both local and remote refs exist, the local one wins."""
    local_oid = _commit_files(git_repo, {"local.py": b"x = 1\n"})

    # Create a remote ref pointing to a different commit.
    remote_oid = _commit_files(
        git_repo,
        {"remote.py": b"y = 2\n"},
        parents=[local_oid],
    )
    git_repo.references.create("refs/heads/main", local_oid, force=True)
    git_repo.references.create("refs/remotes/origin/main", remote_oid)

    # "main" should resolve to the local branch, not the remote.
    entries = list(list_files(git_repo, "main"))
    paths = {e.path for e in entries}
    assert "local.py" in paths
