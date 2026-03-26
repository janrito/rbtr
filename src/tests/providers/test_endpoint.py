"""Tests for rbtr.providers.endpoint — custom OpenAI-compatible endpoints."""

from collections.abc import Generator
from pathlib import Path

import openai
import openai.types
import pytest
from pytest_mock import MockerFixture

from rbtr.exceptions import RbtrError
from rbtr.providers.endpoint import (
    ModelMetadata,
    _metadata_cache,
    build_model,
    fetch_model_metadata,
    list_endpoints,
    list_models,
    load_endpoint,
    remove_endpoint,
    save_endpoint,
    validate_name,
)

type MockOpenAIModel = openai.types.Model
type MockOpenAIClient = openai.OpenAI

# ── validate_name ────────────────────────────────────────────────────


def test_validate_name_accepts_valid() -> None:
    validate_name("deepinfra")
    validate_name("my-endpoint")
    validate_name("local1")


def test_validate_name_rejects_uppercase() -> None:
    with pytest.raises(RbtrError, match="Invalid endpoint name"):
        validate_name("DeepInfra")


def test_validate_name_rejects_starts_with_digit() -> None:
    with pytest.raises(RbtrError, match="Invalid endpoint name"):
        validate_name("1bad")


def test_validate_name_rejects_empty() -> None:
    with pytest.raises(RbtrError, match="Invalid endpoint name"):
        validate_name("")


def test_validate_name_rejects_spaces() -> None:
    with pytest.raises(RbtrError, match="Invalid endpoint name"):
        validate_name("my endpoint")


# ── Persistence ──────────────────────────────────────────────────────


def test_save_load_remove(creds_path: Path, config_path: Path) -> None:
    assert load_endpoint("test") is None

    save_endpoint("test", "https://api.example.com/v1/", "sk-123")

    ep = load_endpoint("test")
    assert ep is not None
    assert ep.name == "test"
    assert ep.base_url == "https://api.example.com/v1"  # trailing slash stripped
    assert ep.api_key == "sk-123"

    remove_endpoint("test")
    assert load_endpoint("test") is None


def test_list_endpoints(creds_path: Path, config_path: Path) -> None:
    assert list_endpoints() == []

    save_endpoint("bravo", "https://b.com/v1", "k2")
    save_endpoint("alpha", "https://a.com/v1", "k1")

    eps = list_endpoints()
    assert len(eps) == 2
    assert eps[0].name == "alpha"  # sorted
    assert eps[1].name == "bravo"


def test_save_endpoint_no_key(creds_path: Path, config_path: Path) -> None:
    save_endpoint("local", "http://localhost:11434/v1", "")

    ep = load_endpoint("local")
    assert ep is not None
    assert ep.api_key == ""


def test_save_invalid_name_raises() -> None:
    with pytest.raises(RbtrError, match="Invalid endpoint name"):
        save_endpoint("BAD NAME", "https://x.com", "k")


# ── Model metadata ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_metadata_cache() -> Generator[None]:
    """Clear the metadata cache between tests."""
    _metadata_cache.clear()
    yield
    _metadata_cache.clear()


def _mock_models_endpoint(mocker: MockerFixture, models: list[MockOpenAIModel]) -> MockOpenAIClient:
    """Patch `openai.OpenAI` so `client.models.list()` returns *models*.

    `list_models` accesses `.data` on the page object, so we need
    both `.data` and iteration to work.
    """
    page = mocker.MagicMock()
    page.data = models
    page.__iter__ = lambda self: iter(models)
    mock_client = mocker.MagicMock()
    mock_client.models.list.return_value = page
    return mocker.patch("openai.OpenAI", return_value=mock_client)


def test_fetch_model_metadata_extracts_context_length(
    creds_path: Path,
    config_path: Path,
    mocker: MockerFixture,
) -> None:
    """context_length from /models metadata → ModelMetadata."""
    save_endpoint("myep", "http://localhost/v1", "")

    model = mocker.MagicMock()
    model.id = "some-model"
    model.model_extra = {"metadata": {"context_length": 196608}}
    _mock_models_endpoint(mocker, [model])

    result = fetch_model_metadata("myep", "some-model")
    assert result == ModelMetadata(context_window=196608)


def test_fetch_model_metadata_caches_result(
    creds_path: Path,
    config_path: Path,
    mocker: MockerFixture,
) -> None:
    save_endpoint("myep", "http://localhost/v1", "")

    model = mocker.MagicMock()
    model.id = "some-model"
    model.model_extra = {"metadata": {"context_length": 128000}}
    mock_cls = _mock_models_endpoint(mocker, [model])

    fetch_model_metadata("myep", "some-model")
    fetch_model_metadata("myep", "some-model")

    mock_cls.assert_called_once()  # type: ignore[attr-defined]  # only one API call despite two fetches


def test_fetch_model_metadata_returns_none_when_no_context_length(
    creds_path: Path,
    config_path: Path,
    mocker: MockerFixture,
) -> None:
    save_endpoint("myep", "http://localhost/v1", "")

    model = mocker.MagicMock()
    model.id = "some-model"
    model.model_extra = {"metadata": {}}
    _mock_models_endpoint(mocker, [model])

    assert fetch_model_metadata("myep", "some-model") is None


def test_fetch_model_metadata_returns_none_for_unknown_endpoint() -> None:
    assert fetch_model_metadata("nonexistent", "model") is None


def test_list_models_populates_metadata_cache(
    creds_path: Path,
    config_path: Path,
    mocker: MockerFixture,
) -> None:
    """list_models piggybacks metadata caching — no extra HTTP call."""

    save_endpoint("myep", "http://localhost/v1", "")

    m1 = mocker.MagicMock()
    m1.id = "model-a"
    m1.model_extra = {"metadata": {"context_length": 32000}}
    m2 = mocker.MagicMock()
    m2.id = "model-b"
    m2.model_extra = {"metadata": {"context_length": 128000}}
    mock_cls = _mock_models_endpoint(mocker, [m1, m2])

    list_models("myep")

    # Both models should be cached — no additional API call needed.
    assert fetch_model_metadata("myep", "model-a") == ModelMetadata(context_window=32000)
    assert fetch_model_metadata("myep", "model-b") == ModelMetadata(context_window=128000)
    mock_cls.assert_called_once()  # type: ignore[attr-defined]  # mock attrs on type alias


# ── Endpoint provider name ───────────────────────────────────────────


def test_build_model_provider_uses_endpoint_name(
    creds_path: Path,
    config_path: Path,
) -> None:
    """Endpoint provider `name` is the endpoint name, not `'openai'`.

    PydanticAI uses `provider.name` to decide whether thinking parts
    from history belong to the same provider.  Endpoints must not
    claim `'openai'` or cross-provider reasoning IDs leak through.
    """
    save_endpoint("fireworks", "https://api.fireworks.ai/inference/v1", "test-key")
    model = build_model("fireworks", "accounts/fireworks/models/llama-v3-70b")

    assert model._provider.name == "fireworks"  # type: ignore[attr-defined]  # internal pydantic-ai attr
