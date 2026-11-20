"""End-to-end smoke test for `rbtr gc` via subprocess.

Exercises the full CLI → inline-fallback path (no daemon running).
Seeds an Index directly instead of running `rbtr watch` to
keep the test fast and to avoid depending on the GGUF embedding
model during unit runs — the indexing path has its own tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest

from rbtr.domain.models import (
    Repo,
)
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.tests.conftest import make_commit, run_cli


@dataclass(frozen=True)
class TinyRepo:
    path: Path
    c1: str
    c2: str


@pytest.fixture
def tiny_repo(tmp_path: Path) -> TinyRepo:
    """A real git repo with two commits."""
    first = b"""\
def a():
    return 1
"""
    second = b"""\
def a():
    return 2
"""
    path = tmp_path / "repo"
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
    c1 = make_commit(repo, {"a.py": first})
    c2 = make_commit(repo, {"a.py": second}, parents=[c1])
    return TinyRepo(path=path, c1=str(c1), c2=str(c2))


@pytest.fixture
def seeded_repo_both_commits(tiny_repo: TinyRepo, isolated_db: Path) -> Repo:
    """Build both commits and close the store so subprocesses can open it."""
    index = Index.from_config(writable=True)
    path = normalise_repo_path(str(tiny_repo.path))
    build_index(path, tiny_repo.c1, index)
    build_index(path, tiny_repo.c2, index)
    registered = index.register(path)
    index.close()
    return registered


@pytest.fixture
def seeded_repo_first_commit(tiny_repo: TinyRepo, isolated_db: Path) -> Repo:
    """Build only the first commit and close the store."""
    index = Index.from_config(writable=True)
    path = normalise_repo_path(str(tiny_repo.path))
    build_index(path, tiny_repo.c1, index)
    registered = index.register(path)
    index.close()
    return registered


def test_gc_default_keeps_head_drops_unreferenced(
    tiny_repo: TinyRepo,
    seeded_repo_both_commits: Repo,
) -> None:
    """Default `rbtr gc` keeps ref tips (HEAD/main = c2), drops c1."""
    seeded = seeded_repo_both_commits
    r = run_cli(["--json", "gc", "--repo-path", str(tiny_repo.path)])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["kind"] == "gc"
    assert payload["snapshots_dropped"] == 1  # c1 is not a ref tip

    index = Index.from_config(writable=True)
    assert index.has_indexed(at=seeded.at(tiny_repo.c1)) is False
    assert index.has_indexed(at=seeded.at(tiny_repo.c2)) is True


def test_gc_dry_run_changes_nothing(
    tiny_repo: TinyRepo,
    seeded_repo_first_commit: Repo,
) -> None:
    seeded = seeded_repo_first_commit
    r = run_cli(["--json", "gc", "--repo-path", str(tiny_repo.path), "--dry-run"])
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["dry_run"] is True

    index = Index.from_config(writable=True)
    assert index.has_indexed(at=seeded.at(tiny_repo.c1)) is True


def test_gc_watched_only_smoke(
    tiny_repo: TinyRepo,
    seeded_repo_both_commits: Repo,
) -> None:
    """`--keep watched-only` parses and routes; HEAD survives."""
    seeded = seeded_repo_both_commits
    r = run_cli(["--json", "gc", "--repo-path", str(tiny_repo.path), "--keep", "watched-only"])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["kind"] == "gc"
    index = Index.from_config(writable=True)
    assert index.has_indexed(at=seeded.at(tiny_repo.c2)) is True  # HEAD kept


def test_gc_across_every_repo_reclaims_globally(
    tiny_repo: TinyRepo,
    seeded_repo_both_commits: Repo,
) -> None:
    """`--scope all` (no --repo-path) routes through the global path and
    reclaims every indexed repo with the default reclamation."""
    seeded = seeded_repo_both_commits
    r = run_cli(["--json", "gc", "--scope", "all"])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["snapshots_dropped"] == 1  # c1 is not a ref tip
    index = Index.from_config(writable=True)
    assert index.has_indexed(at=seeded.at(tiny_repo.c1)) is False
    assert index.has_indexed(at=seeded.at(tiny_repo.c2)) is True


def test_gc_no_compact_smoke(
    tiny_repo: TinyRepo,
    seeded_repo_both_commits: Repo,
) -> None:
    """`--no-compact` parses (pydantic-settings bool negation) and routes."""
    seeded = seeded_repo_both_commits
    r = run_cli(["--json", "gc", "--repo-path", str(tiny_repo.path), "--no-compact"])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["kind"] == "gc"
    index = Index.from_config(writable=True)
    assert index.has_indexed(at=seeded.at(tiny_repo.c2)) is True  # HEAD kept


def test_gc_across_every_repo_keeps_each_watch_set(
    tiny_repo: TinyRepo,
    seeded_repo_both_commits: Repo,
) -> None:
    """`--scope all --keep watched-only` keeps HEAD and the watch set in every
    repo, so the unwatched tip of `main` at c2 survives as HEAD while the
    older commit goes."""
    seeded = seeded_repo_both_commits
    r = run_cli(["--json", "gc", "--scope", "all", "--keep", "watched-only"])
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["kind"] == "gc"

    index = Index.from_config(writable=True)
    assert index.has_indexed(at=seeded.at(tiny_repo.c2)) is True
    assert index.has_indexed(at=seeded.at(tiny_repo.c1)) is False


@pytest.mark.parametrize(
    "args",
    [
        ["gc", "--keep", "everything", "--keep-refs", "main"],
        ["gc", "--keep", "head-only", "--keep-refs", "main"],
        ["gc", "--scope", "all", "--keep", "head-only"],
        ["gc", "--scope", "all", "--keep-refs", "main"],
    ],
    ids=["everything-and-refs", "head-only-and-refs", "everywhere-head-only", "everywhere-keep"],
)
def test_a_contradictory_retention_drops_nothing(
    args: list[str], tiny_repo: TinyRepo, seeded_repo_both_commits: Repo
) -> None:
    """`--keep-refs` is the set kept, so naming a set as well says it
    twice; and the sets naming one repo's refs cannot be asked of every
    repo. Either way the run is refused with both commits still
    indexed."""
    r = run_cli([*args, "--repo-path", str(tiny_repo.path)])

    assert r.returncode == 2, r.stdout
    index = Index.from_config(writable=False)
    try:
        for sha in (tiny_repo.c1, tiny_repo.c2):
            at = seeded_repo_both_commits.at(sha)
            assert index.has_indexed(at=at) is True
    finally:
        index.close()
