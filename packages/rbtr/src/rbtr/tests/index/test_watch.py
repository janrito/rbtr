"""Behaviour of `unwatch_refs`, `remove_stale_refs` and `forget_stale_repos`.

Three operations over the same registered repos: one drops refs a caller
names, one asks a live repo which of its watched refs git still resolves,
the last asks whether a repo is there at all. Driven against a real
in-memory store and real git repos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.domain.models import Scope
from rbtr.errors import RbtrError
from rbtr.index.store import IndexStore
from rbtr.index.watch import forget_stale_repos, remove_stale_refs, unwatch_refs


@pytest.fixture
def vanished_repo(tmp_path: Path) -> str:
    """A path that was never created: a checkout someone deleted."""
    return str(tmp_path / "gone")


@pytest.fixture
def stale_store(
    store: IndexStore, fake_repo: str, second_repo: str, vanished_repo: str
) -> IndexStore:
    """Two live repos watching a dead branch each, beside a registration
    whose checkout is gone."""
    with store.session() as ws:
        first = ws.register_repo(fake_repo)
        second = ws.register_repo(second_repo)
        vanished = ws.register_repo(vanished_repo)
        ws.add_watched_refs(first, ["HEAD", "main", "gone-branch"])
        ws.add_watched_refs(second, ["HEAD", "deleted-branch"])
        ws.add_watched_refs(vanished, ["HEAD", "main"])
    return store


# ── remove_stale_refs ────────────────────────────────────────────────


def test_a_repo_path_scopes_removal_to_that_repo(
    stale_store: IndexStore, fake_repo: str, second_repo: str
) -> None:
    removed = remove_stale_refs(
        stale_store, repo_path=fake_repo, scope=Scope.WORKSPACE, dry_run=False
    )

    assert removed == {fake_repo: ["gone-branch"]}
    assert stale_store.list_watched_refs(stale_store.resolve_repo(fake_repo)) == ["HEAD", "main"]
    assert "deleted-branch" in stale_store.list_watched_refs(stale_store.resolve_repo(second_repo))


def test_every_repo_scope_covers_every_live_repo(
    stale_store: IndexStore, fake_repo: str, second_repo: str
) -> None:
    """Each removed ref says which repo it came from."""
    removed = remove_stale_refs(stale_store, repo_path=fake_repo, scope=Scope.ALL, dry_run=False)

    assert removed == {fake_repo: ["gone-branch"], second_repo: ["deleted-branch"]}
    assert stale_store.list_watched_refs(stale_store.resolve_repo(fake_repo)) == ["HEAD", "main"]
    assert stale_store.list_watched_refs(stale_store.resolve_repo(second_repo)) == ["HEAD"]


def test_a_vanished_repo_is_left_alone(
    stale_store: IndexStore, fake_repo: str, vanished_repo: str
) -> None:
    """Git cannot answer for it, so every ref it watches would look
    stale. Whether the repo itself should go is the other question."""
    removed = remove_stale_refs(stale_store, repo_path=fake_repo, scope=Scope.ALL, dry_run=False)

    assert vanished_repo not in removed
    assert stale_store.list_watched_refs(stale_store.resolve_repo(vanished_repo)) == [
        "HEAD",
        "main",
    ]


def test_head_is_never_stale(store: IndexStore, fake_repo: str) -> None:
    with store.session() as ws:
        repo_id = ws.register_repo(fake_repo)
        ws.add_watched_refs(repo_id, ["HEAD"])

    assert remove_stale_refs(store, repo_path=fake_repo, scope=Scope.ALL, dry_run=False) == {}
    assert store.list_watched_refs(repo_id) == ["HEAD"]


def test_removing_refs_dry_run_reports_without_writing(
    stale_store: IndexStore, fake_repo: str, second_repo: str
) -> None:
    removed = remove_stale_refs(stale_store, repo_path=fake_repo, scope=Scope.ALL, dry_run=True)

    assert removed == {fake_repo: ["gone-branch"], second_repo: ["deleted-branch"]}
    assert "gone-branch" in stale_store.list_watched_refs(stale_store.resolve_repo(fake_repo))


# ── unwatch_refs ─────────────────────────────────────────────────────


def test_named_refs_are_dropped_and_the_rest_kept(stale_store: IndexStore, fake_repo: str) -> None:
    removed = unwatch_refs(stale_store, repo_path=fake_repo, refs=["main"])

    assert removed == {fake_repo: ["main"]}
    assert stale_store.list_watched_refs(stale_store.resolve_repo(fake_repo)) == [
        "HEAD",
        "gone-branch",
    ]


def test_a_repo_that_was_never_indexed_is_left_unregistered(
    stale_store: IndexStore, vanished_repo: str, tmp_path: Path
) -> None:
    """Unwatching in an unknown repo drops nothing and registers nothing."""
    unknown = str(tmp_path / "never-indexed")

    assert unwatch_refs(stale_store, repo_path=unknown, refs=["main"]) == {}
    assert stale_store.get_repo_id(unknown) is None


def test_head_is_refused_before_anything_is_dropped(
    stale_store: IndexStore, fake_repo: str
) -> None:
    """A request naming HEAD alongside others changes nothing at all."""
    with pytest.raises(RbtrError, match="HEAD"):
        unwatch_refs(stale_store, repo_path=fake_repo, refs=["main", "HEAD"])

    assert stale_store.list_watched_refs(stale_store.resolve_repo(fake_repo)) == [
        "HEAD",
        "gone-branch",
        "main",
    ]


# ── forget_stale_repos ───────────────────────────────────────────────


def test_only_repos_whose_path_is_gone_are_forgotten(
    stale_store: IndexStore, vanished_repo: str, fake_repo: str
) -> None:
    gone = forget_stale_repos(stale_store, dry_run=False)

    assert [repo.repo_path for repo in gone] == [vanished_repo]
    assert stale_store.get_repo_id(vanished_repo) is None
    assert stale_store.get_repo_id(fake_repo) is not None


def test_forgetting_repos_dry_run_reports_without_writing(
    stale_store: IndexStore, vanished_repo: str
) -> None:
    gone = forget_stale_repos(stale_store, dry_run=True)

    assert [repo.repo_path for repo in gone] == [vanished_repo]
    assert stale_store.get_repo_id(vanished_repo) is not None
