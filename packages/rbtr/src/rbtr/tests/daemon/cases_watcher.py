"""Scenarios for `rbtr.daemon.watcher.poll`.

Cases return a `WatcherScenario` — pure declarative data
describing which repos exist, their state, and the expected
watcher output.  Shared fixtures in `test_watcher.py` convert
scenarios into real git repos and a real `IndexStore`.

Worktree scenarios live in `case_watcher_worktree.py`.

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
class WatcherScenario:
    """Declarative description of a watcher test scenario."""

    repos: list[RepoSpec] = field(default_factory=list)
    # Map repo name -> zero-based commit index to mark as indexed.
    # A name missing from this map means no ``indexed_commits`` row.
    indexed_at: dict[str, int] = field(default_factory=dict)
    # Repo names whose current HEAD should appear in ``poll``'s
    # output.  The fixture resolves names to ``StaleHead`` objects.
    expected_stale: list[str] = field(default_factory=list)


# ── No-stale scenarios ───────────────────────────────────────────────


def case_empty_store() -> WatcherScenario:
    """No repos registered."""
    return WatcherScenario()


def case_head_already_indexed() -> WatcherScenario:
    """Registered repo whose HEAD is recorded in `indexed_commits`."""
    return WatcherScenario(
        repos=[RepoSpec(name="r")],
        indexed_at={"r": 0},
    )


def case_registered_path_missing() -> WatcherScenario:
    """Registered path is not a git repo: silently skipped."""
    return WatcherScenario(repos=[RepoSpec(name="gone", commits=0)])


# ── Stale scenarios ──────────────────────────────────────────────────


def case_head_never_indexed() -> WatcherScenario:
    """Registered repo with no `indexed_commits` row at all."""
    return WatcherScenario(
        repos=[RepoSpec(name="r")],
        expected_stale=["r"],
    )


def case_new_commit_since_indexing() -> WatcherScenario:
    """HEAD has advanced past the last indexed SHA."""
    return WatcherScenario(
        repos=[RepoSpec(name="r", commits=2)],
        indexed_at={"r": 0},
        expected_stale=["r"],
    )


def case_mixed_multi_repo() -> WatcherScenario:
    """One repo up to date, one stale: only the stale one reports."""
    return WatcherScenario(
        repos=[RepoSpec(name="fresh"), RepoSpec(name="stale")],
        indexed_at={"fresh": 0},
        expected_stale=["stale"],
    )
