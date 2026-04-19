"""Tests for the per-home invariant pin (`rbtr.home_state`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.home_state import HomeState, StripModeMismatch, ensure_or_pin, load, save

# ── load / save round-trip ───────────────────────────────────────────


def test_load_returns_none_for_unset_home(tmp_path: Path) -> None:
    assert load(tmp_path) is None


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    save(tmp_path, HomeState(strip_docstrings_mode=True))
    assert load(tmp_path) == HomeState(strip_docstrings_mode=True)


def test_save_creates_missing_home_dir(tmp_path: Path) -> None:
    home = tmp_path / "fresh"
    save(home, HomeState(strip_docstrings_mode=False))
    assert (home / "home_state.json").exists()


# ── ensure_or_pin ────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [True, False])
def test_ensure_pins_first_call(tmp_path: Path, mode: bool) -> None:
    ensure_or_pin(tmp_path, mode)
    assert load(tmp_path) == HomeState(strip_docstrings_mode=mode)


@pytest.mark.parametrize("mode", [True, False])
def test_ensure_is_no_op_when_modes_match(tmp_path: Path, mode: bool) -> None:
    ensure_or_pin(tmp_path, mode)
    ensure_or_pin(tmp_path, mode)  # must not raise
    assert load(tmp_path) == HomeState(strip_docstrings_mode=mode)


@pytest.mark.parametrize(
    ("first", "second"),
    [(False, True), (True, False)],
)
def test_ensure_rejects_mode_change(tmp_path: Path, first: bool, second: bool) -> None:
    ensure_or_pin(tmp_path, first)
    with pytest.raises(StripModeMismatch) as exc:
        ensure_or_pin(tmp_path, second)
    assert exc.value.pinned is first
    assert exc.value.requested is second
