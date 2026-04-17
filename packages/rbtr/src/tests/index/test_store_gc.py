"""Behavioural tests for ``IndexStore`` GC primitives.

Scenarios in ``case_store_gc.py`` describe a seeded store and the
per-operation expectations.  Each test function applies one
operation (``drop_commit``, ``sweep_orphan_chunks``,
``sweep_orphan_commits``) and asserts against the relevant
``GcScenario`` expectation fields.
"""

from __future__ import annotations

from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.store import IndexStore
from tests.index.case_store_gc import GcScenario


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_store_gc")
def seeded(scenario: GcScenario) -> tuple[IndexStore, GcScenario]:
    """Build an ``IndexStore`` populated per *scenario*."""
    store = IndexStore()
    for i, path in enumerate(scenario.repo_paths):
        repo_id = store.register_repo(path)
        assert repo_id == i + 1, "cases rely on sequential repo_ids"
        data = scenario.per_repo[i] if i < len(scenario.per_repo) else None
        if data is None:
            continue
        if data.chunks:
            store.insert_chunks(list(data.chunks), repo_id=repo_id)
        for commit, path_, blob in data.snapshots:
            store.insert_snapshot(commit, path_, blob, repo_id=repo_id)
        # Group edges by commit so each insert_edges call is scoped
        # to one commit, matching how build_index writes them.
        by_commit: dict[str, list] = {}
        for commit, edge in data.edges:
            by_commit.setdefault(commit, []).append(edge)
        for commit, edges in by_commit.items():
            store.insert_edges(list(edges), commit, repo_id=repo_id)
        for sha in data.marked:
            store.mark_indexed(repo_id, sha)
    return store, scenario


def test_drop_commit_counts(
    seeded: tuple[IndexStore, GcScenario],
) -> None:
    store, scenario = seeded
    for repo_id, commit, expected in scenario.drop:
        counts = store.drop_commit(repo_id, commit)
        for field_name in ("commits", "snapshots", "edges", "chunks"):
            if field_name in expected:
                got = getattr(counts, field_name)
                assert got == expected[field_name], (
                    f"{field_name}: repo_id={repo_id}, commit={commit!r}, "
                    f"expected {expected[field_name]}, got {got}"
                )


def test_drop_commit_state_after(
    seeded: tuple[IndexStore, GcScenario],
) -> None:
    """Run every drop in order; assert flipped-off commits and survivors."""
    store, scenario = seeded
    for repo_id, commit, expected in scenario.drop:
        store.drop_commit(repo_id, commit)
        for gone in expected.get("gone_commit_ids", []):
            assert store.has_indexed(repo_id, gone) is False, (
                f"expected {gone!r} unindexed after drop"
            )
        surviving = expected.get("surviving_chunk_ids", [])
        if not surviving:
            continue
        remaining_ids = {
            c.id
            for other_repo_id, _p in enumerate(scenario.repo_paths, start=1)
            for c in store.get_chunks("commit_b", repo_id=other_repo_id)
        }
        for sid in surviving:
            assert sid in remaining_ids, (
                f"expected chunk {sid!r} to survive"
            )


def test_drop_commit_leaves_other_repos_alone(
    seeded: tuple[IndexStore, GcScenario],
) -> None:
    """Per-repo scoping: dropping in one repo leaves another's marks.

    Only scenarios with the same sha marked in multiple repos
    exercise this; others skip.
    """
    store, scenario = seeded
    if len(scenario.repo_paths) < 2:
        return

    common = None
    for i, data in enumerate(scenario.per_repo, start=1):
        if data.marked and common is None:
            common = (i, set(data.marked))
        elif data.marked and common is not None:
            common = (common[0], common[1] & set(data.marked))

    if not common or not common[1]:
        return

    source_repo, shas = common
    sha = next(iter(shas))
    store.drop_commit(source_repo, sha)

    for other_id in range(1, len(scenario.repo_paths) + 1):
        if other_id == source_repo:
            continue
        if sha in scenario.per_repo[other_id - 1].marked:
            assert store.has_indexed(other_id, sha) is True, (
                f"other repo's mark on {sha!r} should survive"
            )


def test_sweep_orphan_chunks_returns_expected_count(
    seeded: tuple[IndexStore, GcScenario],
) -> None:
    store, scenario = seeded
    for repo_id, expected in scenario.sweep_chunks:
        actual = store.sweep_orphan_chunks(repo_id)
        assert actual == expected, f"repo_id={repo_id}"


def test_sweep_orphan_commits_counts_and_survivors(
    seeded: tuple[IndexStore, GcScenario],
) -> None:
    store, scenario = seeded
    for repo_id, expected_counts, survivors in scenario.sweep_commits:
        counts = store.sweep_orphan_commits(repo_id)
        for key, value in expected_counts.items():
            got = getattr(counts, key)
            assert got == value, (
                f"{key}: repo_id={repo_id}, expected {value}, got {got}"
            )
        remaining = [sha for sha, _at in store.list_indexed_commits(repo_id)]
        assert sorted(remaining) == sorted(survivors), (
            f"repo_id={repo_id} survivors"
        )
