"""Tests for token usage tracking and formatting."""

import pytest

from rbtr.usage import (
    DEFAULT_CONTEXT_WINDOW,
    MessageCountStatus,
    SessionUsage,
    ThresholdStatus,
    format_cost,
    format_tokens,
)

# ── SessionUsage ─────────────────────────────────────────────────────


def test_record_run_accumulates() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=1000, output_tokens=500)
    u.record_run(input_tokens=2000, output_tokens=1000)
    assert u.input_tokens == 3000
    assert u.output_tokens == 1500
    assert u.last_input_tokens == 2000


def test_record_run_tracks_cost() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=1000, output_tokens=500, cost=0.05)
    assert u.total_cost == pytest.approx(0.05)


def test_record_run_accumulates_cost() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=100, output_tokens=50, cost=0.01)
    u.record_run(input_tokens=200, output_tokens=100, cost=0.02)
    assert u.total_cost == pytest.approx(0.03)


def test_record_run_zero_cost_default() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=1000, output_tokens=500)
    assert u.total_cost == 0.0


def test_record_run_cache_tokens() -> None:
    u = SessionUsage()
    u.record_run(
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        cache_write_tokens=100,
    )
    assert u.cache_read_tokens == 200
    assert u.cache_write_tokens == 100


def test_record_run_updates_context_window() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=100, output_tokens=50, context_window=200_000)
    assert u.context_window == 200_000


def test_record_run_none_context_resets_to_default() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=100, output_tokens=50, context_window=200_000)
    assert u.context_window_known is True
    u.record_run(input_tokens=200, output_tokens=100, context_window=None)
    assert u.context_window == DEFAULT_CONTEXT_WINDOW
    assert u.context_window_known is False


def test_default_context_window() -> None:
    u = SessionUsage()
    assert u.context_window == DEFAULT_CONTEXT_WINDOW


def test_context_used_pct() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=100_000, output_tokens=50, context_window=200_000)
    assert u.context_used_pct == pytest.approx(50.0)


def test_context_used_pct_zero_window() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=100, output_tokens=50, context_window=0)
    assert u.context_used_pct == 0.0


def test_threshold_ok_under_70() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=50_000, output_tokens=100, context_window=200_000)
    assert u.threshold_status == ThresholdStatus.OK


def test_threshold_ok_at_boundary() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=139_999, output_tokens=100, context_window=200_000)
    assert u.threshold_status == ThresholdStatus.OK


def test_threshold_warning_at_70() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=140_000, output_tokens=100, context_window=200_000)
    assert u.threshold_status == ThresholdStatus.WARNING


def test_threshold_warning_under_90() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=160_000, output_tokens=100, context_window=200_000)
    assert u.threshold_status == ThresholdStatus.WARNING


def test_threshold_critical_at_90() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=180_000, output_tokens=100, context_window=200_000)
    assert u.threshold_status == ThresholdStatus.CRITICAL


def test_threshold_critical_above_90() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=190_000, output_tokens=100, context_window=200_000)
    assert u.threshold_status == ThresholdStatus.CRITICAL


def test_record_run_increments_message_count() -> None:
    u = SessionUsage()
    assert u.message_count == 0
    u.record_run(input_tokens=100, output_tokens=50)
    assert u.message_count == 1
    u.record_run(input_tokens=200, output_tokens=100)
    assert u.message_count == 2


def test_message_count_status_ok() -> None:
    u = SessionUsage()
    for _ in range(25):
        u.record_run(input_tokens=10, output_tokens=5)
    assert u.message_count_status == MessageCountStatus.OK


def test_message_count_status_warning() -> None:
    u = SessionUsage()
    for _ in range(26):
        u.record_run(input_tokens=10, output_tokens=5)
    assert u.message_count_status == MessageCountStatus.WARNING


def test_message_count_status_critical() -> None:
    u = SessionUsage()
    for _ in range(51):
        u.record_run(input_tokens=10, output_tokens=5)
    assert u.message_count_status == MessageCountStatus.CRITICAL


def test_context_window_known_flag() -> None:
    u = SessionUsage()
    assert u.context_window_known is False
    u.record_run(input_tokens=100, output_tokens=50, context_window=200_000)
    assert u.context_window_known is True
    u.record_run(input_tokens=100, output_tokens=50)
    assert u.context_window_known is False
    assert u.context_window == DEFAULT_CONTEXT_WINDOW


def test_reset_clears_everything() -> None:
    u = SessionUsage()
    u.record_run(input_tokens=1000, output_tokens=500, cost=0.05, context_window=200_000)
    u.reset()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cache_read_tokens == 0
    assert u.cache_write_tokens == 0
    assert u.total_cost == 0.0
    assert u.last_input_tokens == 0
    assert u.message_count == 0
    assert u.context_window == DEFAULT_CONTEXT_WINDOW
    assert u.context_window_known is False


# ── format_tokens ────────────────────────────────────────────────────


def test_format_tokens_small() -> None:
    assert format_tokens(42) == "42"
    assert format_tokens(999) == "999"


def test_format_tokens_thousands() -> None:
    assert format_tokens(1000) == "1.0k"
    assert format_tokens(12345) == "12.3k"
    assert format_tokens(99999) == "100.0k"


def test_format_tokens_hundred_thousands() -> None:
    assert format_tokens(100_000) == "100k"
    assert format_tokens(500_000) == "500k"


def test_format_tokens_millions() -> None:
    assert format_tokens(1_000_000) == "1.0M"
    assert format_tokens(2_500_000) == "2.5M"


# ── format_cost ──────────────────────────────────────────────────────


def test_format_cost_small() -> None:
    assert format_cost(0.0001) == "$0.0001"
    assert format_cost(0.005) == "$0.0050"


def test_format_cost_cents() -> None:
    assert format_cost(0.05) == "$0.05"
    assert format_cost(0.99) == "$0.99"


def test_format_cost_dollars() -> None:
    assert format_cost(1.50) == "$1.50"
    assert format_cost(12.34) == "$12.34"
