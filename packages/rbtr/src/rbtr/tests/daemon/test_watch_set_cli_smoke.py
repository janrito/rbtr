"""End-to-end smoke for `watch`, `unwatch` and `forget` via subprocess.

Drives the inline (no-daemon) path and reads the store afterwards:
refs join and leave the watch set, repos are registered and
forgotten, and a contradictory run changes neither.
"""

from __future__ import annotations

import json
from pathlib import Path

import pygit2
import pytest

from rbtr.git import normalise_repo_path
from rbtr.index.index import Index
from rbtr.tests.conftest import run_cli


@pytest.fixture
def repo_with_stale_watch(fake_repo: str, isolated_db: Path) -> str:
    """A real repo whose watch set holds HEAD, main, and a deleted branch."""
    index = Index.from_config(writable=True)
    repo = index.register(fake_repo)
    with index.session(repo) as ws:
        ws.add_watched_refs(["HEAD", "main", "gone-branch"])
    index.close()
    return fake_repo


def test_fresh_repo_indexes_end_to_end(
    git_repo: pygit2.Repository, isolated_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh repo indexes end-to-end via the real CLI.

    Resilience acceptance for the Aim: opening a new repo gets it
    indexed.  `watch --no-daemon` with embedding off runs the build
    inline and loads no embedding model — GPU-free, and no daemon to contend
    with the rest of the suite (the daemon-start race is covered by
    `test_start_concurrency`).
    """
    monkeypatch.setenv("RBTR_EMBED", "false")
    repo = str(git_repo.workdir)
    result = run_cli(["watch", "--no-daemon", "--repo-path", repo])
    assert result.returncode == 0, result.stderr

    index = Index.from_config(writable=True)
    try:
        registered = index.find_repo(normalise_repo_path(repo))
        assert registered is not None, "repo not registered"
        commits = index.indexed_snapshots(registered)
        assert len(commits) == 1, "HEAD not indexed"
        head = registered.at(commits[0].snapshot_sha)
        counts = index.chunk_counts_for_snapshot(at=head)
        assert counts.total > 0, "no symbols extracted"
        assert counts.embedded == 0, "embedded with embedding off"
    finally:
        index.close()


def test_unwatch_stale_drops_unresolvable_refs(repo_with_stale_watch: str) -> None:
    r = run_cli(["unwatch", "--stale", "--no-daemon", "--repo-path", repo_with_stale_watch])
    assert r.returncode == 0, r.stderr

    index = Index.from_config(writable=True)
    try:
        repo = index.find_repo(repo_with_stale_watch)
        assert repo is not None
        watched = index.watched_refs(repo)
    finally:
        index.close()
    assert "gone-branch" not in watched  # unresolvable → pruned
    assert "main" in watched  # resolvable → kept
    assert "HEAD" in watched  # always kept


@pytest.fixture
def second_repo_with_stale_watch(repo_with_stale_watch: str, second_repo: str) -> str:
    """A second live repo with its own dead branch, beside
    `repo_with_stale_watch`."""
    index = Index.from_config(writable=True)
    repo = index.register(second_repo)
    with index.session(repo) as ws:
        ws.add_watched_refs(["HEAD", "deleted-branch"])
    index.close()
    return second_repo


@pytest.fixture
def vanished_repo(repo_with_stale_watch: str, tmp_path: Path) -> str:
    """A registered path that was never created on disk."""
    path = str(tmp_path / "gone")
    index = Index.from_config(writable=True)
    index.register(path)
    index.close()
    return path


def test_unwatch_stale_everywhere_covers_every_live_repo(
    repo_with_stale_watch: str, second_repo_with_stale_watch: str, vanished_repo: str
) -> None:
    """One invocation covers both live repos, says so as JSON on stdout,
    and leaves the repo whose checkout is gone to `rbtr forget`."""
    r = run_cli(["--json", "unwatch", "--stale", "--scope", "all", "--no-daemon"])
    assert r.returncode == 0, r.stderr

    payload = json.loads(r.stdout)
    assert payload["kind"] == "unwatch"
    assert payload["removed"] == {
        repo_with_stale_watch: ["gone-branch"],
        second_repo_with_stale_watch: ["deleted-branch"],
    }

    index = Index.from_config(writable=True)
    try:
        assert index.find_repo(vanished_repo) is not None
        assert index.watched_refs(index.repo(repo_with_stale_watch)) == [
            "HEAD",
            "main",
        ]
        assert index.watched_refs(index.repo(second_repo_with_stale_watch)) == ["HEAD"]
    finally:
        index.close()


def test_unwatch_dry_run_removes_nothing(
    repo_with_stale_watch: str, second_repo_with_stale_watch: str
) -> None:
    r = run_cli(["--json", "unwatch", "--stale", "--scope", "all", "--no-daemon", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["dry_run"] is True

    index = Index.from_config(writable=True)
    try:
        assert "gone-branch" in index.watched_refs(index.repo(repo_with_stale_watch))
    finally:
        index.close()


@pytest.fixture
def head_only_repo(fake_repo: str, isolated_db: Path) -> str:
    """A real repo registered with HEAD as its only watched ref."""
    index = Index.from_config(writable=True)
    repo = index.register(fake_repo)
    with index.session(repo) as ws:
        ws.add_watched_refs(["HEAD"])
    index.close()
    return fake_repo


def test_forget_drops_a_head_only_repo(head_only_repo: str) -> None:
    """`rbtr forget` drops a repo watching nothing but HEAD."""
    r = run_cli(["forget", "--no-daemon", "--repo-path", head_only_repo])
    assert r.returncode == 0, r.stderr
    index = Index.from_config(writable=True)
    try:
        assert index.find_repo(normalise_repo_path(head_only_repo)) is None
    finally:
        index.close()


def test_forget_stale_forgets_vanished_repos(tmp_path: Path, isolated_db: Path) -> None:
    """`rbtr forget --stale` forgets a repo whose path is gone, needing no
    current repo of its own."""
    gone = str(tmp_path / "gone")  # never created on disk
    index = Index.from_config(writable=True)
    index.register(gone)
    index.close()

    r = run_cli(["forget", "--stale", "--no-daemon"])
    assert r.returncode == 0, r.stderr

    index = Index.from_config(writable=True)
    try:
        assert index.find_repo(gone) is None
    finally:
        index.close()


@pytest.mark.parametrize(
    "args",
    [
        ["unwatch", "main", "--stale"],
        ["unwatch"],
        ["unwatch", "main", "--scope", "all"],
        ["forget", "--stale"],
    ],
    ids=["named-and-stale", "neither", "named-everywhere", "stale-and-named-repo"],
)
def test_a_contradictory_run_changes_nothing(
    args: list[str], repo_with_stale_watch: str, vanished_repo: str
) -> None:
    """Asking for two things at once is refused before any write.

    Each invocation names a ref *and* the rule that finds refs, or
    names neither, or asks one repo's question of every repo. The
    watch set and the registrations must be exactly as they were.
    """
    r = run_cli([*args, "--no-daemon", "--repo-path", repo_with_stale_watch])

    assert r.returncode == 2, r.stdout
    index = Index.from_config(writable=False)
    try:
        assert index.watched_refs(index.repo(repo_with_stale_watch)) == [
            "HEAD",
            "gone-branch",
            "main",
        ]
        assert index.find_repo(vanished_repo) is not None
    finally:
        index.close()
