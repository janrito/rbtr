"""Behavioural tests for ``register_repo`` and ``list_repos``."""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.store import IndexStore
from tests.index.case_store_repos import RepoSequence


@fixture
@parametrize_with_cases("sequence", cases="tests.index.case_store_repos")
def after(sequence: RepoSequence) -> tuple[IndexStore, RepoSequence, list[int]]:
    store = IndexStore()
    issued = [store.register_repo(p) for p in sequence.calls]
    return store, sequence, issued


def test_register_repo_issues_expected_ids(
    after: tuple[IndexStore, RepoSequence, list[int]],
) -> None:
    _store, sequence, issued = after
    assert issued == sequence.expected_ids


def test_list_repos_matches_expected(
    after: tuple[IndexStore, RepoSequence, list[int]],
) -> None:
    store, sequence, _ = after
    assert sorted(store.list_repos()) == sorted(sequence.expected_listing)
