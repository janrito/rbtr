"""Tests for `rbtr.git.resolve_build_ref` and `resolve_read_ref`."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.errors import RbtrError
from rbtr.git import resolve_build_ref, resolve_read_ref

from .conftest import SampleRepo


@pytest.fixture
def sample_repo_path(sample_repo: SampleRepo) -> str:
    """Filesystem path of *sample_repo*'s working directory.

    `sample_repo.repo.path` is the `.git/` directory; `resolve_read_ref`
    expects the working dir.
    """
    return str(Path(sample_repo.repo.path).parent)


# ── resolve_build_ref ────────────────────────────────────────────────


@pytest.mark.parametrize("ref", ["HEAD", "main"])
def test_resolve_build_ref_returns_base_sha(sample_repo: SampleRepo, ref: str) -> None:
    assert resolve_build_ref(sample_repo.repo, ref) == str(sample_repo.base)


def test_resolve_build_ref_resolves_branch(sample_repo: SampleRepo) -> None:
    assert resolve_build_ref(sample_repo.repo, "feature") == str(sample_repo.head)


def test_resolve_build_ref_resolves_full_sha(sample_repo: SampleRepo) -> None:
    sha = str(sample_repo.head)
    assert resolve_build_ref(sample_repo.repo, sha) == sha


def test_resolve_build_ref_raises_on_unknown(sample_repo: SampleRepo) -> None:
    with pytest.raises(RbtrError, match="no_such_branch"):
        resolve_build_ref(sample_repo.repo, "no_such_branch")


# ── resolve_read_ref: SHA short-circuit ──────────────────────────────


def test_sha_short_circuits_without_repo(tmp_path: Path) -> None:
    sha = "deadbeef" * 5  # 40 hex chars
    bogus = str(tmp_path / "no-such-repo")
    assert resolve_read_ref(bogus, sha) == sha


@pytest.mark.parametrize(
    "not_a_sha",
    [
        "abc123",  # too short
        "z" * 40,  # right length but non-hex
        "main",
    ],
)
def test_non_sha_does_not_short_circuit(tmp_path: Path, not_a_sha: str) -> None:
    """A non-SHA against an unopenable repo with no fallback
    returns None, proving the SHA branch didn't echo the input.

    `HEAD` excluded because it has its own fallback path covered
    by the latest-indexed tests below.
    """
    bogus = str(tmp_path / "no-such-repo")
    assert resolve_read_ref(bogus, not_a_sha) is None


# ── resolve_read_ref: pygit2 path ────────────────────────────────────


def test_resolves_head_via_pygit2(sample_repo: SampleRepo, sample_repo_path: str) -> None:
    assert resolve_read_ref(sample_repo_path, "HEAD") == str(sample_repo.base)


def test_resolves_branch_via_pygit2(sample_repo: SampleRepo, sample_repo_path: str) -> None:
    assert resolve_read_ref(sample_repo_path, "feature") == str(sample_repo.head)


def test_returns_none_for_unknown_ref(sample_repo_path: str) -> None:
    assert resolve_read_ref(sample_repo_path, "no_such_branch") is None


# ── resolve_read_ref: latest-indexed fallback ────────────────────────


def test_head_falls_back_to_latest_indexed(tmp_path: Path) -> None:
    bogus = str(tmp_path / "no-such-repo")
    sentinel = "f" * 40
    assert resolve_read_ref(bogus, "HEAD", latest_indexed=lambda: sentinel) == sentinel


def test_fallback_only_fires_for_head(tmp_path: Path) -> None:
    bogus = str(tmp_path / "no-such-repo")
    out = resolve_read_ref(bogus, "main", latest_indexed=lambda: "f" * 40)
    assert out is None


def test_fallback_skipped_when_repo_is_openable(
    sample_repo: SampleRepo, sample_repo_path: str
) -> None:
    """An open repo always wins over the fallback callable."""
    out = resolve_read_ref(sample_repo_path, "HEAD", latest_indexed=lambda: "f" * 40)
    assert out == str(sample_repo.base)


def test_fallback_returns_none_when_no_indexed_commits(tmp_path: Path) -> None:
    bogus = str(tmp_path / "no-such-repo")
    assert resolve_read_ref(bogus, "HEAD", latest_indexed=lambda: None) is None


def test_no_fallback_returns_none_for_missing_repo(tmp_path: Path) -> None:
    bogus = str(tmp_path / "no-such-repo")
    assert resolve_read_ref(bogus, "HEAD") is None
