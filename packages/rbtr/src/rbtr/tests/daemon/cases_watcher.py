"""Scenarios for `rbtr.daemon.watcher.poll_watched`.

Cases return a `WatcherScenario` — pure declarative data
describing which repos exist, what each watches, their indexed
state, and the expected watcher output.  Shared fixtures in
`test_watcher.py` convert scenarios into real git repos and a
real `IndexStore`.

Worktree scenarios live in `cases_watcher_worktree.py`.

Cases hold no I/O, no helpers, no references to pygit2.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepoSpec:
    """A repository to build for a watcher scenario.

    `commits == 0` means do not create a repo on disk — used to
    exercise the path where `list_repos` returns a path the
    filesystem no longer has.
    `register == False` means do not call `ws.register_repo`
    on this repo; used to prove the watcher only considers
    registered repos.
    """

    name: str
    commits: int = 1
    register: bool = True


@dataclass(frozen=True)
class ExpectedTarget:
    """One expected `WatchedTarget`, resolved by the fixture.

    `ref` is the symbolic watched name; `None` means a bare-SHA
    watch whose ref equals the SHA at `sha_index`.  `sha_index`
    indexes the repo's commit SHAs (`-1` = tip).
    """

    repo: str
    ref: str | None
    sha_index: int = -1


@dataclass(frozen=True)
class WatcherScenario:
    """Declarative description of a `poll_watched` test scenario."""

    repos: list[RepoSpec] = field(default_factory=list)
    # Map repo name -> zero-based commit index to mark as indexed.
    indexed_at: dict[str, int] = field(default_factory=dict)
    # Map repo name -> symbolic refs to watch (e.g. ["HEAD"]).
    watched: dict[str, list[str]] = field(default_factory=dict)
    # Map repo name -> commit indices to watch as bare SHAs.
    watched_sha_at: dict[str, list[int]] = field(default_factory=dict)
    # Expected `poll_watched` output, in order.
    expected: list[ExpectedTarget] = field(default_factory=list)


# ── No-target scenarios ──────────────────────────────────────────────


def case_empty_store() -> WatcherScenario:
    """No repos registered."""
    return WatcherScenario()


def case_head_already_indexed() -> WatcherScenario:
    """Watched HEAD is recorded in `indexed_commits`."""
    return WatcherScenario(
        repos=[RepoSpec(name="r")],
        indexed_at={"r": 0},
        watched={"r": ["HEAD"]},
    )


def case_registered_path_missing() -> WatcherScenario:
    """Registered path is not a git repo: the watched ref is skipped."""
    return WatcherScenario(
        repos=[RepoSpec(name="gone", commits=0)],
        watched={"gone": ["HEAD"]},
    )


def case_unresolvable_ref_skipped() -> WatcherScenario:
    """A deleted/unknown branch is skipped; other watched refs still report."""
    return WatcherScenario(
        repos=[RepoSpec(name="r")],
        watched={"r": ["nonexistent-branch", "HEAD"]},
        expected=[ExpectedTarget(repo="r", ref="HEAD")],
    )


def case_bare_sha_one_shot() -> WatcherScenario:
    """A watched bare SHA, once indexed, never reports again (one-shot)."""
    return WatcherScenario(
        repos=[RepoSpec(name="r", commits=2)],
        indexed_at={"r": 0},
        watched_sha_at={"r": [0]},
    )


# ── Target scenarios ─────────────────────────────────────────────────


def case_head_never_indexed() -> WatcherScenario:
    """Watched HEAD with no `indexed_commits` row at all."""
    return WatcherScenario(
        repos=[RepoSpec(name="r")],
        watched={"r": ["HEAD"]},
        expected=[ExpectedTarget(repo="r", ref="HEAD")],
    )


def case_new_commit_since_indexing() -> WatcherScenario:
    """Watched HEAD (moving) has advanced past the last indexed SHA."""
    return WatcherScenario(
        repos=[RepoSpec(name="r", commits=2)],
        indexed_at={"r": 0},
        watched={"r": ["HEAD"]},
        expected=[ExpectedTarget(repo="r", ref="HEAD")],
    )


def case_bare_sha_unindexed() -> WatcherScenario:
    """A watched bare SHA that is not indexed reports with ref == sha."""
    return WatcherScenario(
        repos=[RepoSpec(name="r")],
        watched_sha_at={"r": [0]},
        expected=[ExpectedTarget(repo="r", ref=None, sha_index=0)],
    )


def case_head_and_branch_dedupe() -> WatcherScenario:
    """`HEAD` and `main` resolve to the same commit: one target, not two."""
    return WatcherScenario(
        repos=[RepoSpec(name="r")],
        watched={"r": ["HEAD", "main"]},
        expected=[ExpectedTarget(repo="r", ref="HEAD")],
    )


def case_mixed_multi_repo() -> WatcherScenario:
    """One repo up to date, one stale: only the stale one reports."""
    return WatcherScenario(
        repos=[RepoSpec(name="fresh"), RepoSpec(name="stale")],
        indexed_at={"fresh": 0},
        watched={"fresh": ["HEAD"], "stale": ["HEAD"]},
        expected=[ExpectedTarget(repo="stale", ref="HEAD")],
    )
