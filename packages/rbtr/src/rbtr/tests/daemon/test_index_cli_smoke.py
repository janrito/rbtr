"""End-to-end smoke for the watch-set commands via subprocess.

Exercises the inline (no-daemon) prune path: watched refs that no
longer resolve are removed; HEAD and resolvable refs are kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import pygit2
import pytest

from rbtr.domain.models import SnapshotRef
from rbtr.git import normalise_repo_path
from rbtr.index.store import IndexStore
from rbtr.tests.conftest import run_cli


@pytest.fixture
def repo_with_stale_watch(fake_repo: str, isolated_db: Path) -> str:
    """A real repo whose watch set holds HEAD, main, and a deleted branch."""
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(fake_repo)
        ws.add_watched_refs(repo_id, ["HEAD", "main", "gone-branch"])
    store.close()
    return fake_repo


def test_fresh_repo_indexes_end_to_end(git_repo: pygit2.Repository, isolated_db: Path) -> None:
    """A fresh repo indexes end-to-end via the real CLI.

    Resilience acceptance for the Aim: opening a new repo gets it
    indexed.  `--no-daemon --no-embed` runs the build inline and
    loads no embedding model — GPU-free, and no daemon to contend
    with the rest of the suite (the daemon-start race is covered by
    `test_start_concurrency`).
    """
    repo = str(git_repo.workdir)
    result = run_cli(["watch", "--no-daemon", "--no-embed", "--repo-path", repo])
    assert result.returncode == 0, result.stderr

    store = IndexStore.from_config(writable=True)
    try:
        repo_id = store.get_repo_id(normalise_repo_path(repo))
        assert repo_id is not None, "repo not registered"
        commits = store.list_indexed_snapshots(repo_id)
        assert len(commits) == 1, "HEAD not indexed"
        head = SnapshotRef(repo_id=repo_id, snapshot_sha=commits[0][0])
        assert store.chunk_counts_for_snapshot(at=head).total > 0, "no symbols extracted"
    finally:
        store.close()


def test_unwatch_stale_drops_unresolvable_refs(repo_with_stale_watch: str) -> None:
    r = run_cli(["unwatch", "--stale", "--no-daemon", "--repo-path", repo_with_stale_watch])
    assert r.returncode == 0, r.stderr

    store = IndexStore.from_config(writable=True)
    try:
        repo_id = store.get_repo_id(repo_with_stale_watch)
        assert repo_id is not None
        watched = store.list_watched_refs(repo_id)
    finally:
        store.close()
    assert "gone-branch" not in watched  # unresolvable → pruned
    assert "main" in watched  # resolvable → kept
    assert "HEAD" in watched  # always kept


@pytest.fixture
def second_repo_with_stale_watch(repo_with_stale_watch: str, second_repo: str) -> str:
    """A second live repo with its own dead branch, beside
    `repo_with_stale_watch`."""
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(second_repo)
        ws.add_watched_refs(repo_id, ["HEAD", "deleted-branch"])
    store.close()
    return second_repo


@pytest.fixture
def vanished_repo(repo_with_stale_watch: str, tmp_path: Path) -> str:
    """A registered path that was never created on disk."""
    path = str(tmp_path / "gone")
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        ws.register_repo(path)
    store.close()
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

    store = IndexStore.from_config(writable=True)
    try:
        assert store.get_repo_id(vanished_repo) is not None
        assert store.list_watched_refs(store.resolve_repo(repo_with_stale_watch)) == [
            "HEAD",
            "main",
        ]
        assert store.list_watched_refs(store.resolve_repo(second_repo_with_stale_watch)) == ["HEAD"]
    finally:
        store.close()


def test_unwatch_dry_run_removes_nothing(
    repo_with_stale_watch: str, second_repo_with_stale_watch: str
) -> None:
    r = run_cli(["--json", "unwatch", "--stale", "--scope", "all", "--no-daemon", "--dry-run"])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["dry_run"] is True

    store = IndexStore.from_config(writable=True)
    try:
        assert "gone-branch" in store.list_watched_refs(store.resolve_repo(repo_with_stale_watch))
    finally:
        store.close()


@pytest.fixture
def head_only_repo(fake_repo: str, isolated_db: Path) -> str:
    """A real repo registered with HEAD as its only watched ref."""
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        repo_id = ws.register_repo(fake_repo)
        ws.add_watched_refs(repo_id, ["HEAD"])
    store.close()
    return fake_repo


def test_forget_drops_a_head_only_repo(head_only_repo: str) -> None:
    """`rbtr forget` drops a repo watching nothing but HEAD."""
    r = run_cli(["forget", "--no-daemon", "--repo-path", head_only_repo])
    assert r.returncode == 0, r.stderr
    store = IndexStore.from_config(writable=True)
    try:
        assert store.get_repo_id(normalise_repo_path(head_only_repo)) is None
    finally:
        store.close()


def test_forget_stale_forgets_vanished_repos(tmp_path: Path, isolated_db: Path) -> None:
    """`rbtr forget --stale` forgets a repo whose path is gone, needing no
    current repo of its own."""
    gone = str(tmp_path / "gone")  # never created on disk
    store = IndexStore.from_config(writable=True)
    with store.session() as ws:
        ws.register_repo(gone)
    store.close()

    r = run_cli(["forget", "--stale", "--no-daemon"])
    assert r.returncode == 0, r.stderr

    store = IndexStore.from_config(writable=True)
    try:
        assert store.get_repo_id(gone) is None
    finally:
        store.close()


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
    store = IndexStore.from_config(writable=False)
    try:
        assert store.list_watched_refs(store.resolve_repo(repo_with_stale_watch)) == [
            "HEAD",
            "gone-branch",
            "main",
        ]
        assert store.get_repo_id(vanished_repo) is not None
    finally:
        store.close()
