"""Behavioural tests for `rbtr.daemon.watcher.poll_watched` and `poll_worktree`.

`poll_watched(store)` resolves each repo's `watched_refs` and
returns those whose SHA is not recorded in `indexed_commits`.
Scenarios in `cases_watcher.py`.

`poll_worktree(store)` returns repos whose working tree is
dirty and not yet indexed.  Scenarios in
`cases_watcher_worktree.py`.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pygit2
from pytest_cases import fixture, parametrize_with_cases

from rbtr.daemon.watcher import DirtyWorktree, WatchedTarget, poll_watched, poll_worktree
from rbtr.git import worktree_tree_sha
from rbtr.index.store import IndexStore

from .cases_watcher import WatcherScenario
from .cases_watcher_worktree import WorktreeScenario


@dataclass(frozen=True)
class BuiltRepo:
    """A real git repo on disk plus the SHAs of its commits."""

    path: str
    shas: list[str]


@fixture
def sig() -> pygit2.Signature:
    """Author/committer identity shared by every built commit."""
    return pygit2.Signature("t", "t@t.t")


@fixture
@parametrize_with_cases("s")
def scenario(s: WatcherScenario) -> WatcherScenario:
    """Expose each case as the `scenario` fixture for downstream use."""
    return s


@fixture
def built_repos(
    scenario: WatcherScenario,
    sig: pygit2.Signature,
    tmp_path: Path,
) -> dict[str, BuiltRepo]:
    """Materialise every `RepoSpec` in *scenario* to disk."""
    out: dict[str, BuiltRepo] = {}
    for spec in scenario.repos:
        repo_dir = tmp_path / spec.name
        repo_dir.mkdir()
        shas: list[str] = []
        if spec.commits > 0:
            repo = pygit2.init_repository(str(repo_dir), bare=False, initial_head="main")
            for i in range(spec.commits):
                tb = repo.TreeBuilder()
                tb.insert(
                    f"f{i}.py",
                    repo.create_blob(f"# commit {i}\n".encode()),
                    pygit2.GIT_FILEMODE_BLOB,
                )
                parents = [repo.head.target] if not repo.head_is_unborn else []
                new = repo.create_commit("refs/heads/main", sig, sig, f"c{i}", tb.write(), parents)
                shas.append(str(new))
        out[spec.name] = BuiltRepo(path=str(repo_dir), shas=shas)
    return out


@fixture
def seeded_store(
    scenario: WatcherScenario,
    built_repos: dict[str, BuiltRepo],
) -> Generator[IndexStore]:
    """Register every repo, apply recorded marks, and seed watched refs."""
    store = IndexStore(writable=True)
    for spec in scenario.repos:
        if not spec.register:
            continue
        built = built_repos[spec.name]
        with store.session() as ws:
            repo_id = ws.register_repo(built.path)
            marked_index = scenario.indexed_at.get(spec.name)
            if marked_index is not None:
                ws.mark_indexed(repo_id, built.shas[marked_index])
            symbolic = scenario.watched.get(spec.name, [])
            bare = [built.shas[i] for i in scenario.watched_sha_at.get(spec.name, [])]
            ws.add_watched_refs(repo_id, [*symbolic, *bare])
    yield store
    store.close()


@fixture
def expected_targets(
    scenario: WatcherScenario,
    built_repos: dict[str, BuiltRepo],
) -> list[WatchedTarget]:
    """Compute the expected `poll_watched` output from the scenario."""
    out: list[WatchedTarget] = []
    for e in scenario.expected:
        built = built_repos[e.repo]
        sha = built.shas[e.sha_index]
        out.append(WatchedTarget(repo_path=built.path, ref=e.ref or sha, sha=sha))
    return out


def test_poll_watched(
    seeded_store: IndexStore,
    expected_targets: list[WatchedTarget],
) -> None:
    assert poll_watched(seeded_store) == expected_targets


# ── poll_worktree ───────────────────────────────────────────────────


@fixture
@parametrize_with_cases("s", cases=".cases_watcher_worktree")
def wt_scenario(s: WorktreeScenario) -> WorktreeScenario:
    return s


@fixture
def wt_repo(
    wt_scenario: WorktreeScenario,
    sig: pygit2.Signature,
    tmp_path: Path,
) -> str:
    """Create a real git repo on disk; return its path.

    If `repo_exists` is False, returns a path that doesn't exist.
    """
    repo_dir = tmp_path / "wt_repo"
    if not wt_scenario.repo_exists:
        return str(repo_dir)
    repo_dir.mkdir()
    repo = pygit2.init_repository(str(repo_dir), bare=False, initial_head="main")
    # Initial commit with a tracked file.
    (repo_dir / "main.py").write_text("x = 1\n")
    repo.index.add("main.py")
    repo.index.write()
    tree = repo.index.write_tree()
    repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
    # Dirty the file if the scenario asks.
    if wt_scenario.dirty_file:
        (repo_dir / "main.py").write_text("x = 2\n")
    return str(repo_dir)


@fixture
def wt_store(
    wt_scenario: WorktreeScenario,
    wt_repo: str,
) -> Generator[IndexStore]:
    """Store with the repo registered and optionally tree-SHA-indexed."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(wt_repo)
        # Mark HEAD indexed (so poll() doesn't interfere).
        if wt_scenario.repo_exists:
            head_sha = pygit2.Repository(wt_repo).head.target
            ws.mark_indexed(repo_id, str(head_sha))
        if wt_scenario.tree_sha_indexed:
            tree_sha = worktree_tree_sha(wt_repo)
            if tree_sha is not None:
                ws.mark_indexed(repo_id, tree_sha)
    yield store
    store.close()


def test_poll_worktree(
    wt_scenario: WorktreeScenario,
    wt_store: IndexStore,
    wt_repo: str,
) -> None:
    result = poll_worktree(wt_store)

    if wt_scenario.expected_dirty:
        assert len(result) == 1
        dirty = result[0]
        assert isinstance(dirty, DirtyWorktree)
        assert dirty.repo_path == wt_repo
        # tree_sha should be a valid 40-char hex SHA.
        assert len(dirty.tree_sha) == 40
        # And it should match the current worktree_tree_sha.
        assert dirty.tree_sha == worktree_tree_sha(wt_repo)
    else:
        assert result == []
