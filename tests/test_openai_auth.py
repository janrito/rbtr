"""Tests for OpenAI API key persistence via creds."""

from pathlib import Path

from rbtr.creds import creds


def test_save_load_clear_api_key(creds_path: Path) -> None:
    assert creds.openai_api_key == ""

    creds.update(openai_api_key="sk-test123")
    assert creds.openai_api_key == "sk-test123"

    creds.update(openai_api_key="")
    assert creds.openai_api_key == ""


def test_empty_key_is_falsy(creds_path: Path) -> None:
    creds.update(openai_api_key="")
    assert not creds.openai_api_key
