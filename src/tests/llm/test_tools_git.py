"""Tests for git tools — diff, changed_files, commit_log."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pygit2
import pytest
from pydantic_ai import RunContext

from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.git import changed_files, commit_log, diff
from rbtr.models import BranchTarget
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

from .ctx import tool_ctx

# ── Git tool helpers ─────────────────────────────────────────────────


def _make_repo_two_commits(tmp: str) -> tuple[pygit2.Repository, str, str]:
    """Create a repo with two commits: initial a.py, then modify a.py + add b.py."""
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")

    b1 = repo.create_blob(b"x = 1\n")
    tb1 = repo.TreeBuilder()
    tb1.insert("a.py", b1, pygit2.GIT_FILEMODE_BLOB)
    c1 = repo.create_commit("refs/heads/main", sig, sig, "initial", tb1.write(), [])
    repo.set_head("refs/heads/main")

    b2 = repo.create_blob(b"x = 2\ny = 3\n")
    b3 = repo.create_blob(b"def helper():\n    pass\n")
    tb2 = repo.TreeBuilder()
    tb2.insert("a.py", b2, pygit2.GIT_FILEMODE_BLOB)
    tb2.insert("b.py", b3, pygit2.GIT_FILEMODE_BLOB)
    c2 = repo.create_commit("refs/heads/feature", sig, sig, "add b and change a", tb2.write(), [c1])

    return repo, str(c1), str(c2)


def _git_state(repo: pygit2.Repository) -> EngineState:
    state = EngineState(repo=repo, owner="o", repo_name="r")
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=0,
    )
    return state


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[pygit2.Repository, str, str]:
    """Two-commit git repo for git tool tests."""
    return _make_repo_two_commits(str(tmp_path))


@pytest.fixture
def ctx(git_repo: tuple[pygit2.Repository, str, str], store: SessionStore) -> RunContext[AgentDeps]:
    """RunContext wired to the standard git-tool repo."""
    repo, _, _ = git_repo
    return tool_ctx(_git_state(repo), store)


# ── diff ─────────────────────────────────────────────────────────────


def test_diff_shows_both_changed_files(ctx: RunContext[AgentDeps]) -> None:
    """Default diff (base → head) shows both a.py and b.py changes."""
    result = diff(ctx)
    assert "a.py" in result
    assert "b.py" in result
    assert "files changed" in result


def test_diff_single_ref_shows_commit(store: SessionStore) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, c2 = _make_repo_two_commits(tmp)
        ctx = tool_ctx(_git_state(repo), store)
        result = diff(ctx, ref=c2[:8])
        assert "files changed" in result
        assert "a.py" in result


def test_diff_range_syntax(store: SessionStore) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, c1, c2 = _make_repo_two_commits(tmp)
        ctx = tool_ctx(_git_state(repo), store)
        result = diff(ctx, ref=f"{c1[:8]}..{c2[:8]}")
        assert "files changed" in result


def test_diff_bad_ref(ctx: RunContext[AgentDeps]) -> None:
    result = diff(ctx, ref="nonexistent")
    assert "Cannot resolve ref" in result


def test_diff_no_target(store: SessionStore) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        state = EngineState(repo=repo, owner="o", repo_name="r")
        ctx = tool_ctx(state, store)
        result = diff(ctx)
        assert "No diff target" in result


# ── changed_files ────────────────────────────────────────────────────


def test_changed_files_lists_modified_and_added(ctx: RunContext[AgentDeps]) -> None:
    """changed_files returns both modified and added files."""
    result = changed_files(ctx)
    assert "a.py" in result  # modified
    assert "b.py" in result  # added
    assert "Changed files (2)" in result


def test_changed_files_includes_deleted(store: SessionStore) -> None:
    """Deleted files appear in the changed list."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")

        # Base: two files.
        b1 = repo.create_blob(b"x = 1\n")
        b2 = repo.create_blob(b"y = 2\n")
        tb1 = repo.TreeBuilder()
        tb1.insert("keep.py", b1, pygit2.GIT_FILEMODE_BLOB)
        tb1.insert("remove.py", b2, pygit2.GIT_FILEMODE_BLOB)
        c1 = repo.create_commit("refs/heads/main", sig, sig, "init", tb1.write(), [])
        repo.set_head("refs/heads/main")

        # Head: only keep.py remains.
        tb2 = repo.TreeBuilder()
        tb2.insert("keep.py", b1, pygit2.GIT_FILEMODE_BLOB)
        repo.create_commit("refs/heads/feature", sig, sig, "delete", tb2.write(), [c1])

        ctx = tool_ctx(_git_state(repo), store)
        result = changed_files(ctx)
        assert "remove.py" in result
        assert "keep.py" not in result  # unchanged — should not appear


def test_changed_files_identical_branches(store: SessionStore) -> None:
    """Identical branches report no changes."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        ctx = tool_ctx(_git_state(repo), store)
        result = changed_files(ctx)
        assert "No files changed" in result


def test_changed_files_no_target(store: SessionStore) -> None:
    """No review target returns message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        state = EngineState(repo=repo, owner="o", repo_name="r")
        ctx = tool_ctx(state, store)
        result = changed_files(ctx)
        assert "No diff target" in result


# ── commit_log ───────────────────────────────────────────────────────


def test_commit_log_shows_message(ctx: RunContext[AgentDeps]) -> None:
    result = commit_log(ctx)
    assert "add b and change a" in result


def test_commit_log_identical_branches(store: SessionStore) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        ctx = tool_ctx(_git_state(repo), store)
        result = commit_log(ctx)
        assert "No commits" in result or "identical" in result.lower()


def test_commit_log_no_target(store: SessionStore) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")

        state = EngineState(repo=repo, owner="o", repo_name="r")
        ctx = tool_ctx(state, store)
        result = commit_log(ctx)
        assert "No diff target" in result


# ── diff edge cases ─────────────────────────────────────────────────


def test_diff_initial_commit(store: SessionStore) -> None:
    """Diffing the initial commit (no parent) returns a message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        blob = repo.create_blob(b"x = 1\n")
        tb = repo.TreeBuilder()
        tb.insert("a.py", blob, pygit2.GIT_FILEMODE_BLOB)
        c = repo.create_commit("refs/heads/main", sig, sig, "init", tb.write(), [])
        repo.set_head("refs/heads/main")

        state = _git_state(repo)
        ctx = tool_ctx(state, store)
        result = diff(ctx, ref=str(c)[:8])
        assert "no parent" in result or "initial commit" in result


def test_diff_bad_range_refs(ctx: RunContext[AgentDeps]) -> None:
    """Unresolvable refs in range syntax returns error."""
    result = diff(ctx, ref="badref1..badref2")
    assert "Cannot resolve ref" in result


def test_diff_truncation(config_path: Path, store: SessionStore) -> None:
    """Large diffs are limited at max_lines."""
    from rbtr.config import config as cfg

    cfg.tools.max_lines = 5  # tiny limit for test

    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")

        b1 = repo.create_blob(b"")
        tb1 = repo.TreeBuilder()
        tb1.insert("a.py", b1, pygit2.GIT_FILEMODE_BLOB)
        c1 = repo.create_commit("refs/heads/main", sig, sig, "init", tb1.write(), [])
        repo.set_head("refs/heads/main")

        # Make a big change
        big = ("x = 1\n" * 200).encode()
        b2 = repo.create_blob(big)
        tb2 = repo.TreeBuilder()
        tb2.insert("a.py", b2, pygit2.GIT_FILEMODE_BLOB)
        repo.create_commit("refs/heads/feature", sig, sig, "big", tb2.write(), [c1])

        state = _git_state(repo)
        ctx = tool_ctx(state, store)
        result = diff(ctx)
        assert "... limited" in result
        assert "offset=" in result


# ── diff with pattern ────────────────────────────────────────────────


def test_diff_pattern_filters_to_single_file(ctx: RunContext[AgentDeps]) -> None:
    """pattern='a.py' shows only that file's diff."""
    result = diff(ctx, pattern="a.py")
    assert "a.py" in result
    assert "b.py" not in result
    assert "1 files changed" in result


def test_diff_pattern_empty_shows_full(ctx: RunContext[AgentDeps]) -> None:
    """Empty pattern (default) shows the full diff as before."""
    result = diff(ctx)
    assert "a.py" in result
    assert "b.py" in result
    assert "2 files changed" in result


def test_diff_pattern_nonexistent(ctx: RunContext[AgentDeps]) -> None:
    """Nonexistent pattern returns empty diff."""
    result = diff(ctx, pattern="nonexistent.py")
    assert "0 files changed" in result


def test_diff_pattern_with_single_ref(store: SessionStore) -> None:
    """pattern also works in single-ref mode."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, c2 = _make_repo_two_commits(tmp)
        ctx = tool_ctx(_git_state(repo), store)
        result = diff(ctx, pattern="a.py", ref=c2[:8])
        assert "a.py" in result
        assert "b.py" not in result


def test_diff_pattern_with_range(store: SessionStore) -> None:
    """pattern also works with range syntax."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, c1, c2 = _make_repo_two_commits(tmp)
        ctx = tool_ctx(_git_state(repo), store)
        result = diff(ctx, pattern="b.py", ref=f"{c1[:8]}..{c2[:8]}")
        assert "b.py" in result
        assert "a.py" not in result


def test_diff_glob_star(ctx: RunContext[AgentDeps]) -> None:
    """Glob `*.py` shows only Python file diffs."""
    result = diff(ctx, pattern="*.py")
    assert "a.py" in result
    assert "b.py" in result
    assert "2 files changed" in result


def test_diff_glob_no_match(ctx: RunContext[AgentDeps]) -> None:
    """Glob with no matching files returns empty diff."""
    result = diff(ctx, pattern="*.rs")
    assert "0 files changed" in result


# ── commit_log edge cases ────────────────────────────────────────────


def test_commit_log_bad_refs(store: SessionStore) -> None:
    """Unresolvable branch refs returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")

        state = EngineState(repo=repo, owner="o", repo_name="r")
        state.review_target = BranchTarget(
            base_branch="main",
            head_branch="nonexistent",
            base_commit="main",
            head_commit="nonexistent",
            updated_at=0,
        )
        ctx = tool_ctx(state, store)
        result = commit_log(ctx)
        assert "Cannot resolve ref" in result


def test_commit_log_truncation(config_path: Path, store: SessionStore) -> None:
    """Long commit log is limited at max_results."""
    from rbtr.config import config as cfg

    cfg.tools.max_results = 2  # tiny limit

    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        c = repo.create_commit("refs/heads/main", sig, sig, "base", tree, [])
        repo.set_head("refs/heads/main")

        # Add 5 commits on feature
        parent_id = c
        for i in range(5):
            parent_id = repo.create_commit(
                "refs/heads/feature",
                sig,
                sig,
                f"commit {i}",
                tree,
                [parent_id],
            )

        state = _git_state(repo)
        ctx = tool_ctx(state, store)
        result = commit_log(ctx)
        assert "... limited" in result
        assert "offset=" in result
