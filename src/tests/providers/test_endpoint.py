"""Tests for rbtr.providers.endpoint — custom OpenAI-compatible endpoints."""

from pathlib import Path

import pytest

from rbtr.exceptions import RbtrError
from rbtr.providers.endpoint import (
    list_endpoints,
    load_endpoint,
    remove_endpoint,
    save_endpoint,
    validate_name,
)

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
