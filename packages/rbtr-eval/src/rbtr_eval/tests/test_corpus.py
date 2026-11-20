"""`corpus_refs` names the snapshots the eval measures.

Builds a real git repo into a real `Index`: `head_sha` reads git,
so neither can be faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.config import config
from rbtr.domain.models import Repo
from rbtr.errors import RbtrError
from rbtr.git import normalise_repo_path, worktree_tree_sha
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.tests.conftest import make_commit
from rbtr_eval.corpus import Corpus, corpus_refs
from rbtr_eval.tests.cases_corpus import HEAD, OTHER, CorpusScenario


@dataclass(frozen=True)
class RepoAtHead:
    """A real git repo on disk with one commit."""

    path: Path
    head: str


@pytest.fixture
def repo_at_head(tmp_path: Path) -> RepoAtHead:
    """A git repo whose single commit adds `a.py`, checked out."""
    a = b"""\
def f():
    pass
"""
    path = tmp_path / "repo"
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
    head = str(make_commit(repo, {"a.py": a}))
    repo.checkout_head(strategy=pygit2.GIT_CHECKOUT_FORCE)  # type: ignore[no-untyped-call]  # pygit2 untyped
    return RepoAtHead(path=path, head=head)


@pytest.fixture
def worktree_sha(repo_at_head: RepoAtHead) -> str:
    """The tree of `repo_at_head`'s checkout with `a.py` edited: a dirty worktree."""
    a = b"""\
def f():
    return 1
"""
    (repo_at_head.path / "a.py").write_bytes(a)
    sha = worktree_tree_sha(str(repo_at_head.path), max_untracked_files=config.max_untracked_files)
    assert sha is not None
    return sha


@pytest.fixture
def index(tmp_path: Path) -> Index:
    """A file-backed, writable index, isolated to this test."""
    return Index(str(tmp_path / "index" / "index.duckdb"), writable=True)


@fixture
@parametrize_with_cases("scenario", cases=".cases_corpus", has_tag="valid")
def valid(scenario: CorpusScenario) -> CorpusScenario:
    return scenario


@fixture
@parametrize_with_cases("scenario", cases=".cases_corpus", has_tag="invalid")
def invalid(scenario: CorpusScenario) -> CorpusScenario:
    return scenario


@pytest.fixture
def valid_repo(
    valid: CorpusScenario, index: Index, repo_at_head: RepoAtHead, worktree_sha: str
) -> Repo:
    """`repo_at_head` registered, with each snapshot the scenario names built."""
    path = normalise_repo_path(str(repo_at_head.path))
    for name in valid.indexed:
        build_index(path, {HEAD: repo_at_head.head, OTHER: worktree_sha}[name], index)
    return index.register(path)


@pytest.fixture
def invalid_repo(
    invalid: CorpusScenario, index: Index, repo_at_head: RepoAtHead, worktree_sha: str
) -> Repo:
    """`repo_at_head` registered, with each snapshot the scenario names built."""
    path = normalise_repo_path(str(repo_at_head.path))
    for name in invalid.indexed:
        build_index(path, {HEAD: repo_at_head.head, OTHER: worktree_sha}[name], index)
    return index.register(path)


def test_corpus_refs_names_each_repos_head(
    valid_repo: Repo, index: Index, repo_at_head: RepoAtHead
) -> None:
    """The corpus is HEAD, given as a `SnapshotRef` per registered repo."""
    assert corpus_refs(index) == Corpus(refs=(valid_repo.at(repo_at_head.head),))


def test_corpus_refs_refuses_an_index_that_is_not_the_corpus(
    invalid: CorpusScenario,
    invalid_repo: Repo,
    index: Index,
    repo_at_head: RepoAtHead,
    worktree_sha: str,
) -> None:
    """Anything but HEAD alone is refused, naming the repo and the SHA."""
    assert invalid.error_match is not None
    error_match = worktree_sha if invalid.error_match == OTHER else invalid.error_match

    with pytest.raises(RbtrError, match=error_match) as excinfo:
        corpus_refs(index)

    assert str(repo_at_head.path) in str(excinfo.value)
