"""Tests for xterm modifyOtherKeys preprocessing."""

import pytest

from rbtr_legacy.tui.key_encoding import preprocess

# ── Shift+Enter / Alt+Enter ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("\x1b[27;2;13~", "\x1b\r", id="shift+enter"),
        pytest.param("\x1b[27;3;13~", "\x1b\r", id="alt+enter"),
    ],
)
def test_enter_with_modifier(raw: str, expected: str) -> None:
    assert preprocess(raw) == expected


# ── Plain Enter (no modifier) ───────────────────────────────────────


def test_plain_enter() -> None:
    """modifyOtherKeys mode 1 doesn't encode plain Enter — passes through."""
    assert preprocess("\r") == "\r"


# ── Escape ───────────────────────────────────────────────────────────


def test_escape() -> None:
    assert preprocess("\x1b[27;1;27~") == "\x1b"


# ── Shift+Tab ────────────────────────────────────────────────────────


def test_shift_tab() -> None:
    assert preprocess("\x1b[27;2;9~") == "\x1b[Z"


# ── Alt+letter ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param("\x1b[27;3;98~", "\x1bb", id="alt+b"),
        pytest.param("\x1b[27;3;102~", "\x1bf", id="alt+f"),
        pytest.param("\x1b[27;3;100~", "\x1bd", id="alt+d"),
    ],
)
def test_alt_letter(raw: str, expected: str) -> None:
    assert preprocess(raw) == expected


# ── Backspace ────────────────────────────────────────────────────────


def test_alt_backspace() -> None:
    assert preprocess("\x1b[27;3;127~") == "\x1b\x7f"


# ── Printable passthrough ────────────────────────────────────────────


def test_printable() -> None:
    assert preprocess("\x1b[27;1;97~") == "a"


# ── Unknown codepoint → stripped ─────────────────────────────────────


def test_unknown_codepoint_stripped() -> None:
    assert preprocess("\x1b[27;1;9999~") == ""


# ── Passthrough (no modifyOtherKeys sequences) ──────────────────────


def test_passthrough_plain_text() -> None:
    assert preprocess("hello world") == "hello world"


def test_passthrough_legacy_escape() -> None:
    assert preprocess("\x1b\r") == "\x1b\r"


# ── Mixed data ───────────────────────────────────────────────────────


def test_mixed_mok_and_plain() -> None:
    """modifyOtherKeys sequences are replaced; surrounding text is preserved."""
    data = "abc\x1b[27;2;13~def"
    assert preprocess(data) == "abc\x1b\rdef"
