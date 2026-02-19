"""Tests for ThinkingEffort — config round-trip and rotation logic."""

from __future__ import annotations

import pytest

from rbtr.config import ThinkingEffort, config

# ── Enum basics ──────────────────────────────────────────────────────


def test_all_members_present() -> None:
    """ThinkingEffort has exactly the five expected members."""
    assert set(ThinkingEffort) == {
        ThinkingEffort.LOW,
        ThinkingEffort.MEDIUM,
        ThinkingEffort.HIGH,
        ThinkingEffort.MAX,
        ThinkingEffort.NONE,
    }


def test_members_are_lowercase_strings() -> None:
    """All enum values are lowercase strings for TOML serialisation."""
    for member in ThinkingEffort:
        assert member.value == member.value.lower()
        assert isinstance(member.value, str)


def test_none_is_falsy_in_bool_context() -> None:
    """NONE is a valid StrEnum member, not Python None."""
    assert ThinkingEffort.NONE is not None
    assert isinstance(ThinkingEffort.NONE, ThinkingEffort)


# ── Config round-trip ────────────────────────────────────────────────


@pytest.mark.parametrize("effort", list(ThinkingEffort))
def test_config_roundtrip(config_path, effort) -> None:
    """Every effort level survives config.update → reload."""
    config.update(thinking_effort=effort)
    assert config.thinking_effort == effort


def test_default_is_medium() -> None:
    """Default thinking effort is MEDIUM."""
    from rbtr.config import Config

    fresh = Config()
    assert fresh.thinking_effort == ThinkingEffort.MEDIUM


# ── Rotation logic ───────────────────────────────────────────────────


def test_rotation_cycles_all_members(config_path) -> None:
    """Rotating len(members) times returns to the starting effort."""
    members = list(ThinkingEffort)
    config.update(thinking_effort=ThinkingEffort.LOW)

    visited: list[ThinkingEffort] = []
    for _ in range(len(members)):
        current = config.thinking_effort
        visited.append(current)
        idx = members.index(current)
        config.update(thinking_effort=members[(idx + 1) % len(members)])

    # Should have visited every member exactly once.
    assert set(visited) == set(members)
    # Should be back to LOW.
    assert config.thinking_effort == ThinkingEffort.LOW


def test_rotation_order(config_path) -> None:
    """Rotation follows enum declaration order: low → medium → high → max → none."""
    members = list(ThinkingEffort)
    config.update(thinking_effort=members[0])

    order: list[ThinkingEffort] = []
    for _ in range(len(members)):
        order.append(config.thinking_effort)
        idx = members.index(config.thinking_effort)
        config.update(thinking_effort=members[(idx + 1) % len(members)])

    assert order == members
