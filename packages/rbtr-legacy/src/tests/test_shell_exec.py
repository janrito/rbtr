"""Tests for `rbtr.shell_exec.truncate_for_agent`."""

from __future__ import annotations

import pytest

from rbtr_legacy.shell_exec import truncate_for_agent


def test_empty_input() -> None:
    assert truncate_for_agent("") == ""


def test_within_both_limits_unchanged() -> None:
    text = "\n".join(f"line{i}" for i in range(10))
    assert truncate_for_agent(text) == text


def test_line_limit_triggers_head_tail() -> None:
    lines = [f"line{i}" for i in range(100)]
    text = "\n".join(lines)
    result = truncate_for_agent(text, max_lines=20, max_bytes=100_000)

    result_lines = result.splitlines()
    # 10 head + 1 divider + 10 tail = 21
    assert len(result_lines) == 21

    # Head preserved.
    assert result_lines[:10] == lines[:10]
    # Tail preserved.
    assert result_lines[11:] == lines[90:]
    # Divider present.
    assert result_lines[10] == "[… 80 lines truncated …]"


def test_byte_limit_triggers_truncation() -> None:
    # 50 lines of 2000 chars each = ~100KB, over 50KB limit.
    lines = ["x" * 2000 for _ in range(50)]
    text = "\n".join(lines)
    result = truncate_for_agent(text, max_lines=2000, max_bytes=51_200)

    assert len(result.encode()) <= 51_200
    assert "[…" in result
    assert "truncated …]" in result


def test_both_limits_line_hits_first() -> None:
    # 500 short lines — line limit of 10 triggers before byte limit.
    lines = [f"L{i}" for i in range(500)]
    text = "\n".join(lines)
    result = truncate_for_agent(text, max_lines=10, max_bytes=100_000)

    result_lines = result.splitlines()
    assert len(result_lines) == 11  # 5 head + 1 divider + 5 tail
    assert result_lines[5] == "[… 490 lines truncated …]"


def test_both_limits_bytes_hits_first() -> None:
    # 20 lines of 5KB each = 100KB. Line limit 2000 won't fire,
    # but byte limit 51_200 will.
    lines = ["y" * 5000 for _ in range(20)]
    text = "\n".join(lines)
    result = truncate_for_agent(text, max_lines=2000, max_bytes=51_200)

    assert len(result.encode()) <= 51_200
    assert "[…" in result


def test_exact_line_limit_unchanged() -> None:
    text = "\n".join(f"line{i}" for i in range(2000))
    assert truncate_for_agent(text, max_lines=2000, max_bytes=1_000_000) == text


def test_exact_byte_limit_unchanged() -> None:
    # Build text that is exactly at the byte boundary.
    text = "a" * 51_200
    assert truncate_for_agent(text, max_lines=2000, max_bytes=51_200) == text


def test_divider_reports_correct_hidden_count() -> None:
    lines = [f"line{i}" for i in range(1000)]
    text = "\n".join(lines)
    result = truncate_for_agent(text, max_lines=100, max_bytes=1_000_000)

    divider_lines = [line for line in result.splitlines() if "truncated" in line]
    assert len(divider_lines) == 1
    assert "900 lines truncated" in divider_lines[0]


@pytest.mark.parametrize("max_bytes", [100, 50, 30])
def test_very_small_byte_limit(max_bytes: int) -> None:
    """Even with an aggressive byte cap the result never exceeds it."""
    lines = [f"line-{i}-content" for i in range(200)]
    text = "\n".join(lines)
    result = truncate_for_agent(text, max_lines=2000, max_bytes=max_bytes)

    assert len(result.encode()) <= max_bytes


def test_single_line_over_byte_limit() -> None:
    """A single very long line that exceeds the byte budget."""
    text = "z" * 100_000
    result = truncate_for_agent(text, max_lines=2000, max_bytes=51_200)

    # Single line can't be split by lines — head and tail are both
    # this one line, so it ends up fully truncated.
    assert len(result.encode()) <= 51_200
