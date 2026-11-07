"""End-to-end cross-repo smoke tests via the real CLI subprocess.

Seeds two repos into a per-test on-disk store (`isolated_db`),
then drives `rbtr` as a subprocess to verify:
- `search --scope all` merges hits from both repos (attributed),
- workspace search/status stays scoped to one repo,
- non-search tools (read-symbol) never surface another repo's data
  even when a symbol name collides across repos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.tests.conftest import make_commit

from ..conftest import run_cli


@dataclass(frozen=True)
class TwoRepos:
    """Two seeded repos sharing the symbol name `shared_fn`."""

    path_a: str
    path_b: str


@pytest.fixture
def two_repos(tmp_path: Path, isolated_db: Path) -> TwoRepos:
    """Build two repos into this test's own store, each at its HEAD.

    Repo A: `alpha_fn` + `shared_fn`.  Repo B: `beta_fn` +
    `shared_fn`.  `shared_fn` collides by name across repos so
    isolation and `(repo_id, id)` keying are both exercised.
    Each repo is built at HEAD so the handler's ref resolution finds it.
    """
    shared = b"""\
def shared_fn():
    pass
"""
    alpha = b"""\
def alpha_fn():
    pass
"""
    beta = b"""\
def beta_fn():
    pass
"""
    index = Index.from_config(writable=True)
    paths: list[str] = []
    for name, files in (
        ("repo_a", {"alpha.py": alpha, "shared.py": shared}),
        ("repo_b", {"beta.py": beta, "shared.py": shared}),
    ):
        repo = pygit2.init_repository(str(tmp_path / name), bare=False, initial_head="main")
        path = normalise_repo_path(repo.workdir)
        sha = str(make_commit(repo, files))
        repo.checkout_head(strategy=pygit2.GIT_CHECKOUT_FORCE)  # type: ignore[no-untyped-call]  # pygit2 untyped
        build_index(path, sha, index)
        paths.append(path)
    index.close()
    return TwoRepos(path_a=paths[0], path_b=paths[1])


def test_search_scope_all_merges_repos(two_repos: TwoRepos) -> None:
    """`search --scope all` returns hits from both repos, attributed."""
    r = run_cli(
        ["--json", "search", "shared_fn", "--scope", "all", "--repo-path", two_repos.path_a]
    )
    assert r.returncode == 0, r.stderr
    # The hit carries repo_path attribution rather than an id; the merge
    # shows as both repos' paths present among the results.
    repo_paths = {hit["repo_path"] for hit in json.loads(r.stdout)["results"]}
    assert repo_paths == {two_repos.path_a, two_repos.path_b}


def test_search_workspace_excludes_other_repo(two_repos: TwoRepos) -> None:
    """A workspace search never surfaces the other repo's chunks."""
    r = run_cli(["--json", "search", "shared_fn", "--repo-path", two_repos.path_b])
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)["results"]
    names = {hit["name"] for hit in hits}
    assert "shared_fn" in names
    # Repo A's unique symbol must not leak into a repo-B workspace search.
    assert "alpha_fn" not in names
    # repo_path is omitted (not null) for workspace hits.
    assert all("repo_path" not in hit for hit in hits)


def test_status_scope_all_lists_both_repos(two_repos: TwoRepos) -> None:
    """`status --scope all` reports exactly the two indexed repos."""
    r = run_cli(["--json", "status", "--scope", "all", "--repo-path", two_repos.path_a])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    repo_paths = {ref["repo_path"] for ref in payload["indexed_refs"]}
    assert repo_paths == {two_repos.path_a, two_repos.path_b}


def test_status_workspace_single_repo(two_repos: TwoRepos) -> None:
    """Workspace status reports only the path's repo."""
    r = run_cli(["--json", "status", "--repo-path", two_repos.path_a])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert all(ref["repo_path"] == two_repos.path_a for ref in payload["indexed_refs"])


def test_read_symbol_isolated_to_repo(two_repos: TwoRepos) -> None:
    """read-symbol for a colliding name returns only the path's repo."""
    r = run_cli(["--json", "read-symbol", "shared_fn", "--repo-path", two_repos.path_a])
    assert r.returncode == 0, r.stderr
    # Both repos hold a `shared_fn`; the DTO carries no id, so isolation
    # shows as a single chunk (a leak would emit the other repo's too).
    chunks = json.loads(r.stdout)["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["name"] == "shared_fn"
