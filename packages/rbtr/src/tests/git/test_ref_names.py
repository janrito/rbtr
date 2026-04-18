"""Tests for ``names_for_commits`` \u2014 reverse-lookup of refs by SHA."""

from __future__ import annotations

from pathlib import Path

import pygit2

from rbtr.git import names_for_commits
from tests.git.conftest import SampleRepo


def test_head_and_branch_names_for_main(sample_repo: SampleRepo) -> None:
    base_sha = str(sample_repo.base)
    result = names_for_commits(sample_repo.repo, [base_sha])
    # main points at `base` and HEAD defaults to main.
    assert set(result[base_sha]) == {"HEAD", "main"}


def test_branch_only_for_non_head_commit(sample_repo: SampleRepo) -> None:
    head_sha = str(sample_repo.head)
    result = names_for_commits(sample_repo.repo, [head_sha])
    # feature points at head; HEAD is on main (the base commit).
    assert result[head_sha] == ["feature"]


def test_empty_list_for_unnamed_commit(sample_repo: SampleRepo) -> None:
    mid_sha = str(sample_repo.mid)
    result = names_for_commits(sample_repo.repo, [mid_sha])
    # Mid has no branch or tag pointing at it.
    assert result[mid_sha] == []


def test_unknown_shas_survive_as_empty_entries(sample_repo: SampleRepo) -> None:
    result = names_for_commits(sample_repo.repo, ["deadbeef" * 5])
    assert result == {"deadbeef" * 5: []}


def test_empty_input_returns_empty_dict(sample_repo: SampleRepo) -> None:
    assert names_for_commits(sample_repo.repo, []) == {}


def test_tag_resolves_to_short_name(sample_repo: SampleRepo) -> None:
    sample_repo.repo.references.create(
        "refs/tags/v1.0", sample_repo.mid
    )
    mid_sha = str(sample_repo.mid)
    result = names_for_commits(sample_repo.repo, [mid_sha])
    assert result[mid_sha] == ["v1.0"]


def test_multiple_shas_in_one_call(sample_repo: SampleRepo) -> None:
    base_sha = str(sample_repo.base)
    head_sha = str(sample_repo.head)
    result = names_for_commits(sample_repo.repo, [base_sha, head_sha])
    assert set(result[base_sha]) == {"HEAD", "main"}
    assert result[head_sha] == ["feature"]


def test_remote_tracking_branch_keeps_remote_prefix(sample_repo: SampleRepo) -> None:
    sample_repo.repo.references.create(
        "refs/remotes/origin/main", sample_repo.head
    )
    head_sha = str(sample_repo.head)
    result = names_for_commits(sample_repo.repo, [head_sha])
    # Local branch and remote-tracking branch both surface; remote
    # keeps its "origin/" prefix so it doesn't collide with the local.
    assert set(result[head_sha]) == {"feature", "origin/main"}


def test_unborn_head_does_not_raise(tmp_path: Path) -> None:
    repo = pygit2.init_repository(str(tmp_path / "empty_repo"))
    assert repo.head_is_unborn
    result = names_for_commits(repo, ["deadbeef" * 5])
    assert result == {"deadbeef" * 5: []}
