"""`corpus_refs` names the snapshots the eval measures.

Seeds a real git repo and a real `IndexStore`: `head_sha` reads git,
so neither can be faked.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygit2
import pytest
from pytest_cases import parametrize_with_cases

from rbtr.domain.models import SnapshotRef
from rbtr.errors import RbtrError
from rbtr.index.store import IndexStore
from rbtr_eval.corpus import corpus_refs
from rbtr_eval.tests.cases_corpus import CorpusScenario
from rbtr_eval.tests.conftest import seed_corpus

# ── Fixtures ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RepoAtHead:
    """A real git repo on disk with one commit."""

    path: Path
    head: str


@pytest.fixture
def repo_at_head(tmp_path: Path) -> RepoAtHead:
    """A git repo whose single commit adds `a.py`."""
    path = tmp_path / "repo"
    repo = pygit2.init_repository(str(path), bare=False, initial_head="main")
    sig = pygit2.Signature("t", "t@t.t")
    tb = repo.TreeBuilder()
    tb.insert("a.py", repo.create_blob(b"def f():\n    pass\n"), pygit2.GIT_FILEMODE_BLOB)
    head = str(repo.create_commit("refs/heads/main", sig, sig, "c1", tb.write(), []))
    return RepoAtHead(path=path, head=head)


@pytest.fixture
def store(tmp_path: Path) -> IndexStore:
    """A file-backed, writable index, isolated to this test."""
    return IndexStore(str(tmp_path / "index" / "index.duckdb"), writable=True)


# ── Tests ────────────────────────────────────────────────────────────


@parametrize_with_cases("scenario", cases=".cases_corpus", has_tag="valid")
def test_corpus_refs_names_each_repos_head(
    scenario: CorpusScenario, store: IndexStore, repo_at_head: RepoAtHead
) -> None:
    """The corpus is HEAD, given as a `SnapshotRef` per registered repo."""
    repo_id = seed_corpus(store, repo_at_head.path, repo_at_head.head, scenario.indexed)

    assert corpus_refs(store) == [SnapshotRef(repo_id=repo_id, snapshot_sha=repo_at_head.head)]


@parametrize_with_cases("scenario", cases=".cases_corpus", has_tag="invalid")
def test_corpus_refs_refuses_an_index_that_is_not_the_corpus(
    scenario: CorpusScenario, store: IndexStore, repo_at_head: RepoAtHead
) -> None:
    """Anything but HEAD alone is refused, naming the repo and the SHA."""
    seed_corpus(store, repo_at_head.path, repo_at_head.head, scenario.indexed)
    assert scenario.error_match is not None

    with pytest.raises(RbtrError, match=scenario.error_match) as excinfo:
        corpus_refs(store)

    assert str(repo_at_head.path) in str(excinfo.value)
