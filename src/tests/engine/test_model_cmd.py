"""Tests for /model — listing, setting, cache behaviour.

Covers the untested paths in ``engine/model.py``: listing with
markers, invalid format, unknown provider, unknown model with
suggestions, cache TTL, and fetch errors.
"""

from __future__ import annotations

import time
from pathlib import Path

from pytest_mock import MockerFixture

from rbtr.engine import Engine, TaskType
from rbtr.exceptions import RbtrError
from rbtr.providers import BuiltinProvider, claude

from .conftest import drain, output_texts

# ── Shared data ──────────────────────────────────────────────────────

_CLAUDE_MODELS = [
    "claude/claude-sonnet-4-20250514",
    "claude/claude-opus-4-20250514",
]

_OPENAI_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]


def _seed_cache(
    engine: Engine,
    *,
    providers: list[tuple[str, list[str]]] | None = None,
    current: str | None = None,
) -> None:
    """Pre-populate model cache so /model doesn't hit real APIs."""
    if providers is None:
        providers = [
            ("claude", list(_CLAUDE_MODELS)),
            ("openai", list(_OPENAI_MODELS)),
        ]
    engine.state.cached_models = providers
    engine.state.models_fetched_at = time.time()
    if current:
        engine.state.model_name = current


# ── Listing ──────────────────────────────────────────────────────────


def test_model_list_no_providers(config_path: Path, engine: Engine) -> None:
    """/model with no providers warns."""
    engine.state.cached_models = []
    engine.state.models_fetched_at = time.time()

    engine.run_task(TaskType.COMMAND, "/model")
    texts = output_texts(drain(engine.events))

    assert any("No LLM connected" in t for t in texts)


def test_model_list_shows_models_with_marker(config_path: Path, engine: Engine) -> None:
    """/model lists models and marks the current one with ◂."""
    _seed_cache(engine, current="claude/claude-sonnet-4-20250514")

    engine.run_task(TaskType.COMMAND, "/model")
    texts = output_texts(drain(engine.events))

    # Current model has marker
    assert any("claude-sonnet-4-20250514" in t and "◂" in t for t in texts)
    # Other model doesn't
    opus_lines = [t for t in texts if "claude-opus-4-20250514" in t]
    assert opus_lines
    assert "◂" not in opus_lines[0]


def test_model_list_empty_provider_shows_usage(config_path: Path, engine: Engine) -> None:
    """/model shows /model <provider>/<id> for providers with empty model lists."""
    _seed_cache(engine, providers=[("ollama", [])])

    engine.run_task(TaskType.COMMAND, "/model")
    texts = output_texts(drain(engine.events))

    assert any("/model ollama/" in t for t in texts)


# ── Setting ──────────────────────────────────────────────────────────


def test_model_set_invalid_format(config_path: Path, engine: Engine) -> None:
    """/model noslash warns about invalid format."""

    engine.run_task(TaskType.COMMAND, "/model noslash")
    texts = output_texts(drain(engine.events))

    assert any("Invalid model format" in t for t in texts)


def test_model_set_unknown_provider(config_path: Path, engine: Engine) -> None:
    """/model fakeprovider/x warns about unknown provider."""
    _seed_cache(engine)

    engine.run_task(TaskType.COMMAND, "/model fakeprovider/x")
    texts = output_texts(drain(engine.events))

    assert any("Unknown provider" in t for t in texts)


def test_model_set_unknown_model_lists_available(
    config_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """/model claude/nonexistent warns but sets anyway (API validates at call time)."""
    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    _seed_cache(engine)

    # Mock the API call so refresh returns the same list
    mocker.patch.object(
        claude.provider,
        "list_models",
        return_value=["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
    )

    engine.run_task(TaskType.COMMAND, "/model claude/nonexistent")
    texts = output_texts(drain(engine.events))

    assert any("not in the known list" in t for t in texts)
    assert any("Model set to" in t for t in texts)


def test_model_set_cache_hit_applies_without_refresh(
    config_path: Path,
    mocker: MockerFixture,
    engine: Engine,
) -> None:
    """/model with a cached model applies immediately (no API call)."""
    _seed_cache(engine)

    # If this fires, it's a bug — we should use the cache.
    spy = mocker.patch.object(
        claude.provider,
        "list_models",
        side_effect=AssertionError("should not be called"),
    )

    engine.run_task(TaskType.COMMAND, "/model claude/claude-opus-4-20250514")
    texts = output_texts(drain(engine.events))

    assert engine.state.model_name == "claude/claude-opus-4-20250514"
    assert any("Model set to" in t for t in texts)
    spy.assert_not_called()


def test_model_set_resets_effort_supported(config_path: Path, engine: Engine) -> None:
    """Setting a model resets effort_supported to None for re-evaluation."""
    engine.state.effort_supported = True
    _seed_cache(engine)

    engine.run_task(TaskType.COMMAND, "/model openai/gpt-4o")
    drain(engine.events)

    assert engine.state.effort_supported is None


# ── Cache behaviour ──────────────────────────────────────────────────


def test_get_models_returns_cached_when_fresh(
    config_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """get_models returns cached data when TTL hasn't expired."""
    from rbtr.engine.model_cmd import get_models

    expected = [("claude", list(_CLAUDE_MODELS))]
    _seed_cache(engine, providers=expected)

    spy = mocker.patch.object(
        claude.provider,
        "list_models",
        side_effect=AssertionError("should not refresh"),
    )

    result = get_models(engine)
    assert result == expected
    spy.assert_not_called()


def test_get_models_force_refreshes(
    config_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """get_models(force=True) refreshes even with a fresh cache."""
    from rbtr.engine.model_cmd import get_models

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    _seed_cache(engine, providers=[("claude", ["claude/old-model"])])

    mocker.patch.object(
        claude.provider,
        "list_models",
        return_value=["claude-sonnet-4-20250514"],
    )

    result = get_models(engine, force=True)
    assert ("claude", ["claude/claude-sonnet-4-20250514"]) in result


def test_get_models_stale_cache_refreshes(
    config_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """get_models refreshes when the cache TTL has expired."""
    from rbtr.engine.model_cmd import get_models

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.cached_models = [("claude", ["claude/old-model"])]
    engine.state.models_fetched_at = time.time() - 600  # 10 min ago (TTL = 5 min)

    mocker.patch.object(
        claude.provider,
        "list_models",
        return_value=["claude-sonnet-4-20250514"],
    )

    result = get_models(engine)
    assert ("claude", ["claude/claude-sonnet-4-20250514"]) in result


def test_fetch_provider_models_error_warns(
    config_path: Path, mocker: MockerFixture, engine: Engine
) -> None:
    """API error during model listing warns and returns empty."""
    from rbtr.engine.model_cmd import get_models

    engine.state.connected_providers.add(BuiltinProvider.CLAUDE)
    engine.state.cached_models = []
    engine.state.models_fetched_at = 0

    mocker.patch.object(
        claude.provider,
        "list_models",
        side_effect=RbtrError("network down"),
    )

    result = get_models(engine, force=True)
    texts = output_texts(drain(engine.events))

    assert ("claude", []) in result
    assert any("Could not list" in t for t in texts)
