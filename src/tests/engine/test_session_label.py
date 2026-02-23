"""Tests for session label generation."""

from __future__ import annotations

import pytest

from rbtr.engine.core import _make_session_label

from .conftest import make_repo_with_file


def test_label_with_branch(tmp_path: pytest.TempPathFactory) -> None:
    """Label includes branch name when on a branch."""
    repo = make_repo_with_file(str(tmp_path))
    label = _make_session_label("acme", "app", repo)
    assert label == "acme/app — main"


def test_label_detached_head(tmp_path: pytest.TempPathFactory) -> None:
    """Label includes short commit hash when HEAD is detached."""
    repo = make_repo_with_file(str(tmp_path))
    commit = repo.head.target
    repo.set_head(commit)  # detach
    label = _make_session_label("acme", "app", repo)
    assert label.startswith("acme/app — ")
    assert len(label.split(" — ")[1]) == 8  # short hash
