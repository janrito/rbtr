"""Tests for /reload — prompt source reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.engine import Engine, TaskType

from .conftest import drain, output_texts


@pytest.fixture
def prompt_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Isolate RBTR_DIR and cwd so file-presence tests are deterministic.

    Returns ``(config_dir, repo_dir)``.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    monkeypatch.setattr("rbtr.engine.reload_cmd.RBTR_DIR", config_dir)
    monkeypatch.chdir(repo_dir)
    return config_dir, repo_dir


# ── Baseline ─────────────────────────────────────────────────────────


def test_builtin_no_extras(prompt_dirs: tuple[Path, Path], engine: Engine) -> None:
    """No overrides — reports built-in system, no append, project not found."""
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("built-in" in t for t in texts)
    assert any("not found" in t for t in texts)
    assert any("refresh" in t.lower() for t in texts)


# ── SYSTEM.md override ───────────────────────────────────────────────


def test_system_override_detected(prompt_dirs: tuple[Path, Path], engine: Engine) -> None:
    """When SYSTEM.md exists, /reload reports it as an override."""
    config_dir, _ = prompt_dirs
    (config_dir / "SYSTEM.md").write_text("Custom persona.")
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("override" in t for t in texts)
    assert not any("built-in" in t for t in texts)


# ── APPEND_SYSTEM.md ─────────────────────────────────────────────────


def test_append_system_detected(prompt_dirs: tuple[Path, Path], engine: Engine) -> None:
    """When APPEND_SYSTEM.md exists, /reload reports it."""
    config_dir, _ = prompt_dirs
    (config_dir / "APPEND_SYSTEM.md").write_text("Extra rules.")
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("append" in t.lower() and "APPEND_SYSTEM" in t for t in texts)


def test_append_system_absent(prompt_dirs: tuple[Path, Path], engine: Engine) -> None:
    """When APPEND_SYSTEM.md is absent, /reload does not mention it."""
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert not any("APPEND_SYSTEM" in t for t in texts)


# ── Project instructions ─────────────────────────────────────────────


def test_project_file_found(prompt_dirs: tuple[Path, Path], engine: Engine) -> None:
    """When AGENTS.md exists in the repo, /reload reports it as found."""
    _, repo_dir = prompt_dirs
    (repo_dir / "AGENTS.md").write_text("Project rules.")
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("AGENTS.md" in t and "not found" not in t for t in texts)


def test_project_file_missing(prompt_dirs: tuple[Path, Path], engine: Engine) -> None:
    """When AGENTS.md does not exist, /reload reports it as not found."""
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("AGENTS.md" in t and "not found" in t for t in texts)


def test_project_file_added_after_start(prompt_dirs: tuple[Path, Path], engine: Engine) -> None:
    """A project file created after startup is detected on next /reload."""
    _, repo_dir = prompt_dirs

    # First reload — missing.
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("not found" in t for t in texts)

    # Create the file.
    (repo_dir / "AGENTS.md").write_text("New rules.")

    # Second reload — found.
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("AGENTS.md" in t and "not found" not in t for t in texts)


def test_custom_project_filenames(
    prompt_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine,
) -> None:
    """Custom project_instructions config is respected."""
    _, repo_dir = prompt_dirs
    (repo_dir / "REVIEW.md").write_text("Review focus.")
    monkeypatch.setattr(
        "rbtr.engine.reload_cmd.config.project_instructions",
        ["REVIEW.md", "MISSING.md"],
    )
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("REVIEW.md" in t and "not found" not in t for t in texts)
    assert any("MISSING.md" in t and "not found" in t for t in texts)


def test_no_project_instructions_configured(
    prompt_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    engine: Engine,
) -> None:
    """When project_instructions is empty, /reload says none configured."""
    monkeypatch.setattr(
        "rbtr.engine.reload_cmd.config.project_instructions", []
    )
    engine.run_task(TaskType.COMMAND, "/reload")
    texts = output_texts(drain(engine.events))
    assert any("none configured" in t for t in texts)
