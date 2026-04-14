"""Tests for workspace discovery and path resolution."""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr_legacy.workspace import resolve_path, workspace_dir

# ── helpers ──────────────────────────────────────────────────────────


def _git_repo(path: Path) -> None:
    """Create a bare-minimum git repo at `path`."""
    path.mkdir(parents=True, exist_ok=True)
    pygit2.init_repository(str(path))


@pytest.fixture(autouse=True)
def _real_workspace_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the conftest mock so these tests exercise the real function."""
    monkeypatch.setattr("rbtr_legacy.workspace.workspace_dir", workspace_dir)
    workspace_dir.cache_clear()


# ── workspace_dir ────────────────────────────────────────────────────


def test_at_git_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    (root / ".rbtr").mkdir()

    monkeypatch.chdir(root)
    assert workspace_dir() == root / ".rbtr"


def test_at_ancestor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    (root / ".rbtr").mkdir()
    (root / "src" / "pkg").mkdir(parents=True)

    monkeypatch.chdir(root / "src" / "pkg")
    assert workspace_dir() == root / ".rbtr"


def test_nearest_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Monorepo: `.rbtr/` in a subdirectory beats one at the root."""
    root = tmp_path / "repo"
    _git_repo(root)
    (root / ".rbtr").mkdir()
    (root / "pkg" / ".rbtr").mkdir(parents=True)
    (root / "pkg" / "sub").mkdir()

    monkeypatch.chdir(root / "pkg" / "sub")
    assert workspace_dir() == root / "pkg" / ".rbtr"


def test_fallback_to_git_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No `.rbtr/` anywhere → defaults to `{git_root}/.rbtr`."""
    root = tmp_path / "repo"
    _git_repo(root)

    monkeypatch.chdir(root)
    result = workspace_dir()
    assert result == root / ".rbtr"
    assert not result.exists()


def test_no_git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert workspace_dir() == tmp_path / ".rbtr"


def test_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _git_repo(root)

    monkeypatch.chdir(root)
    assert workspace_dir() is workspace_dir()


# ── resolve_path ─────────────────────────────────────────────────────


def test_placeholder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    (root / ".rbtr").mkdir()

    monkeypatch.chdir(root)
    assert resolve_path("${WORKSPACE}/index") == root / ".rbtr" / "index"


def test_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert resolve_path("/custom/path") == Path("/custom/path")


def test_relative(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    _git_repo(root)
    (root / ".rbtr").mkdir()

    monkeypatch.chdir(root)
    assert resolve_path(".rbtr/index") == root / ".rbtr" / "index"
