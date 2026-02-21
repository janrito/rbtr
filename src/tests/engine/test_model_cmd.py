"""Tests for /model — listing, setting, cache behaviour.

Covers the untested paths in ``engine/model.py``: listing with
markers, invalid format, unknown provider, unknown model with
suggestions, cache TTL, and fetch errors.
"""

from __future__ import annotations

import time

import pytest

from rbtr.engine import TaskType
from rbtr.exceptions import RbtrError

from .conftest import drain, make_engine, output_texts

# ── Shared data ──────────────────────────────────────────────────────

_CLAUDE_MODELS = [
    "claude/claude-sonnet-4-20250514",
    "claude/claude-opus-4-20250514",
]

_OPENAI_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]


def _seed_cache(session, *, providers=None, current=None):
    """Pre-populate model cache so /model doesn't hit real APIs."""
    if providers is None:
        providers = [
            ("claude", list(_CLAUDE_MODELS)),
            ("openai", list(_OPENAI_MODELS)),
        ]
    session.cached_models = providers
    session.models_fetched_at = time.time()
    if current:
        session.model_name = current


# ── Listing ──────────────────────────────────────────────────────────


def test_model_list_no_providers(config_path) -> None:
    """/model with no providers warns."""
    engine, events, session = make_engine()
    session.cached_models = []
    session.models_fetched_at = time.time()

    engine.run_task(TaskType.COMMAND, "/model")
    texts = output_texts(drain(events))

    assert any("No LLM connected" in t for t in texts)


def test_model_list_shows_models_with_marker(config_path) -> None:
    """/model lists models and marks the current one with ◂."""
    engine, events, session = make_engine()
    _seed_cache(session, current="claude/claude-sonnet-4-20250514")

    engine.run_task(TaskType.COMMAND, "/model")
    texts = output_texts(drain(events))

    # Current model has marker
    assert any("claude-sonnet-4-20250514" in t and "◂" in t for t in texts)
    # Other model doesn't
    opus_lines = [t for t in texts if "claude-opus-4-20250514" in t]
    assert opus_lines
    assert "◂" not in opus_lines[0]


def test_model_list_empty_provider_shows_usage(config_path) -> None:
    """/model shows /model <provider>/<id> for providers with empty model lists."""
    engine, events, session = make_engine()
    _seed_cache(session, providers=[("ollama", [])])

    engine.run_task(TaskType.COMMAND, "/model")
    texts = output_texts(drain(events))

    assert any("/model ollama/" in t for t in texts)


# ── Setting ──────────────────────────────────────────────────────────


def test_model_set_invalid_format(config_path) -> None:
    """/model noslash warns about invalid format."""
    engine, events, _ = make_engine()

    engine.run_task(TaskType.COMMAND, "/model noslash")
    texts = output_texts(drain(events))

    assert any("Invalid model format" in t for t in texts)


def test_model_set_unknown_provider(config_path) -> None:
    """/model fakeprovider/x warns about unknown provider."""
    engine, events, session = make_engine()
    _seed_cache(session)

    engine.run_task(TaskType.COMMAND, "/model fakeprovider/x")
    texts = output_texts(drain(events))

    assert any("Unknown provider" in t for t in texts)


def test_model_set_unknown_model_lists_available(config_path, mocker) -> None:
    """/model claude/nonexistent warns and shows available models."""
    engine, events, session = make_engine()
    session.claude_connected = True
    _seed_cache(session)

    # Mock the API call so refresh returns the same list
    mocker.patch(
        "rbtr.engine.model.claude_provider.list_models",
        return_value=["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
    )

    engine.run_task(TaskType.COMMAND, "/model claude/nonexistent")
    texts = output_texts(drain(events))

    assert any("Unknown model" in t for t in texts)
    assert any("claude-sonnet-4-20250514" in t for t in texts)


def test_model_set_cache_hit_applies_without_refresh(config_path, mocker) -> None:
    """/model with a cached model applies immediately (no API call)."""
    engine, events, session = make_engine()
    _seed_cache(session)

    # If this fires, it's a bug — we should use the cache.
    spy = mocker.patch(
        "rbtr.engine.model.claude_provider.list_models",
        side_effect=AssertionError("should not be called"),
    )

    engine.run_task(TaskType.COMMAND, "/model claude/claude-opus-4-20250514")
    texts = output_texts(drain(events))

    assert session.model_name == "claude/claude-opus-4-20250514"
    assert any("Model set to" in t for t in texts)
    spy.assert_not_called()


def test_model_set_resets_effort_supported(config_path) -> None:
    """Setting a model resets effort_supported to None for re-evaluation."""
    engine, events, session = make_engine()
    session.effort_supported = True
    _seed_cache(session)

    engine.run_task(TaskType.COMMAND, "/model openai/gpt-4o")
    drain(events)

    assert session.effort_supported is None


# ── Cache behaviour ──────────────────────────────────────────────────


def test_get_models_returns_cached_when_fresh(config_path, mocker) -> None:
    """get_models returns cached data when TTL hasn't expired."""
    from rbtr.engine.model import get_models

    engine, _, session = make_engine()
    expected = [("claude", list(_CLAUDE_MODELS))]
    _seed_cache(session, providers=expected)

    spy = mocker.patch(
        "rbtr.engine.model.claude_provider.list_models",
        side_effect=AssertionError("should not refresh"),
    )

    result = get_models(engine)
    assert result == expected
    spy.assert_not_called()


def test_get_models_force_refreshes(config_path, mocker) -> None:
    """get_models(force=True) refreshes even with a fresh cache."""
    from rbtr.engine.model import get_models

    engine, _, session = make_engine()
    session.claude_connected = True
    _seed_cache(session, providers=[("claude", ["claude/old-model"])])

    mocker.patch(
        "rbtr.engine.model.claude_provider.list_models",
        return_value=["claude-sonnet-4-20250514"],
    )

    result = get_models(engine, force=True)
    assert ("claude", ["claude/claude-sonnet-4-20250514"]) in result


def test_get_models_stale_cache_refreshes(config_path, mocker) -> None:
    """get_models refreshes when the cache TTL has expired."""
    from rbtr.engine.model import get_models

    engine, _, session = make_engine()
    session.claude_connected = True
    session.cached_models = [("claude", ["claude/old-model"])]
    session.models_fetched_at = time.time() - 600  # 10 min ago (TTL = 5 min)

    mocker.patch(
        "rbtr.engine.model.claude_provider.list_models",
        return_value=["claude-sonnet-4-20250514"],
    )

    result = get_models(engine)
    assert ("claude", ["claude/claude-sonnet-4-20250514"]) in result


def test_fetch_provider_models_error_warns(config_path, mocker) -> None:
    """API error during model listing warns and returns empty."""
    from rbtr.engine.model import get_models

    engine, events, session = make_engine()
    session.claude_connected = True
    session.cached_models = []
    session.models_fetched_at = 0

    mocker.patch(
        "rbtr.engine.model.claude_provider.list_models",
        side_effect=RbtrError("network down"),
    )

    result = get_models(engine, force=True)
    texts = output_texts(drain(events))

    assert ("claude", []) in result
    assert any("Could not list" in t for t in texts)


# ── Connected providers ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("flags", "expected_providers"),
    [
        ({"claude_connected": True}, ["claude"]),
        ({"chatgpt_connected": True}, ["chatgpt"]),
        ({"openai_connected": True}, ["openai"]),
        (
            {"claude_connected": True, "openai_connected": True},
            ["claude", "openai"],
        ),
    ],
)
def test_connected_providers(config_path, mocker, flags, expected_providers) -> None:
    """_connected_providers returns names of connected providers."""
    from rbtr.engine.model import _connected_providers

    engine, _, session = make_engine()
    for key, value in flags.items():
        setattr(session, key, value)
    mocker.patch("rbtr.engine.model.endpoint_provider.list_endpoints", return_value=[])

    result = _connected_providers(engine)
    assert result == expected_providers
