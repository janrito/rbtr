"""Tests for session label generation."""

from __future__ import annotations

import pygit2
import pytest

from rbtr.engine.core import Engine
from rbtr.engine.setup import _make_session_label
from rbtr.engine.types import TaskType
from tests.helpers import drain

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


def test_review_snapshot_updates_label(repo_engine: Engine) -> None:
    """First /review <ref> sets the session label to the ref."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review feature")
    drain(engine.events)

    assert "feature" in engine.state.session_label
    assert "→" not in engine.state.session_label


def test_review_branch_updates_label(repo_engine: Engine) -> None:
    """First /review <base> <target> sets the session label to base → head."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review main feature")
    drain(engine.events)

    assert "main" in engine.state.session_label
    assert "feature" in engine.state.session_label
    assert "→" in engine.state.session_label


def test_second_review_keeps_label(repo_engine: Engine) -> None:
    """Second /review does not overwrite the label set by the first."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature-a", repo.head.peel(pygit2.Commit))
    repo.branches.local.create("feature-b", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/review feature-a")
    drain(engine.events)
    first_label = engine.state.session_label

    engine.run_task(TaskType.COMMAND, "/review feature-b")
    drain(engine.events)

    assert engine.state.session_label == first_label
    assert "feature-a" in engine.state.session_label


def test_rename_survives_review(repo_engine: Engine) -> None:
    """/session rename is not overwritten by subsequent /review."""
    engine = repo_engine
    repo = engine.state.repo
    assert repo is not None
    repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

    engine.run_task(TaskType.COMMAND, "/session rename my notes")
    drain(engine.events)

    engine.run_task(TaskType.COMMAND, "/review feature")
    drain(engine.events)

    assert engine.state.session_label == "my notes"


def test_label_starts_empty(repo_engine: Engine) -> None:
    """Session label is empty before /review or /session rename."""
    assert repo_engine.state.session_label == ""
