"""Tests for incremental (two-commit) index builds."""

from __future__ import annotations

from pathlib import Path

import pygit2

from rbtr.index.orchestrator import build_index
from rbtr.index.store import IndexStore


def test_incremental_incremental(
    git_repo: pygit2.Repository, store: IndexStore, two_commits: tuple[str, str]
) -> None:
    base_sha, head_sha = two_commits

    # Build base first.
    build_index(git_repo.workdir, base_sha, store, repo_id=1)

    # Incremental update.
    result = build_index(git_repo.workdir, head_sha, store, repo_id=1, base_sha=base_sha)

    assert result.stats.total_files > 0
    assert result.stats.parsed_files >= 1  # At least the changed files.
    assert not result.errors

    # New function should be visible at head.
    chunks = store.get_chunks(head_sha, repo_id=1)
    names = {c.name for c in chunks}
    assert "new_func" in names
    assert "serve" in names


def test_incremental_marks_head_indexed(
    git_repo: pygit2.Repository, store: IndexStore, two_commits: tuple[str, str]
) -> None:
    """Incremental update marks head, leaving base's mark alone."""
    base_sha, head_sha = two_commits
    build_index(git_repo.workdir, base_sha, store, repo_id=1)
    assert store.has_indexed(1, head_sha) is False

    build_index(git_repo.workdir, head_sha, store, repo_id=1, base_sha=base_sha)

    assert store.has_indexed(1, head_sha) is True
    assert store.has_indexed(1, base_sha) is True  # still marked


def test_incremental_preserves_unchanged(
    git_repo: pygit2.Repository, store: IndexStore, two_commits: tuple[str, str]
) -> None:
    base_sha, head_sha = two_commits

    build_index(git_repo.workdir, base_sha, store, repo_id=1)
    result = build_index(git_repo.workdir, head_sha, store, repo_id=1, base_sha=base_sha)

    # Unchanged files should be cached, not re-parsed.
    assert result.stats.skipped_files > 0

    # Old symbols should still be visible at head.
    chunks = store.get_chunks(head_sha, repo_id=1)
    names = {c.name for c in chunks}
    assert "User" in names
    assert "Order" in names


def test_incremental_replaces_head_snapshots_for_reused_ref(
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
    build_index(git_repo.workdir, "main", store, repo_id=1)

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

    build_index(git_repo.workdir, "feature", store, repo_id=1, base_sha="main")
    names_v1 = {c.name for c in store.get_chunks("feature", repo_id=1)}
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

    build_index(git_repo.workdir, "feature", store, repo_id=1, base_sha="main")
    names_v2 = {c.name for c in store.get_chunks("feature", repo_id=1)}
    assert "legacy_only" not in names_v2


def test_incremental_remote_only_head(
    git_repo: pygit2.Repository, store: IndexStore, tmp_path: Path
) -> None:
    """build_index with base_sha works when head is a remote-only branch (PR scenario).

    Reproduces the real-world bug: `/review 900` sets `head_branch`
    to the PR's branch name (e.g. `rewrite-mq`), which only exists
    as `origin/rewrite-mq`.  Without the remote fallback in
    `_resolve_commit`, `build_index with base_sha` throws `RbtrError`.
    """
    base_sha = str(git_repo.head.target)

    # Create a second commit with changes.
    utils_path = tmp_path / "src" / "utils.py"
    utils_path.write_bytes(b"""\
\"\"\"Utility functions.\"\"\"

def helper():
    return 42

def new_func():
    return "new"
""")
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
    build_index(git_repo.workdir, base_sha, store, repo_id=1)
    result = build_index(git_repo.workdir, "feature-branch", store, repo_id=1, base_sha=base_sha)

    assert result.stats.total_chunks > 0
    # Snapshots stored under "feature-branch", queryable by that name.
    chunks = store.get_chunks("feature-branch", repo_id=1)
    assert len(chunks) > 0
    assert any(c.name == "new_func" for c in chunks)
