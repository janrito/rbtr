"""Tests for constants — enforce performance targets and invariants."""

from rbtr.constants import (
    DOUBLE_CTRL_C_WINDOW,
    POLL_INTERVAL,
    REFRESH_PER_SECOND,
    SHELL_COMPLETION_TIMEOUT,
)

# ── Keystroke latency target ─────────────────────────────────────────
#
# Worst-case keystroke-to-display latency is bounded by:
#   POLL_INTERVAL (main loop notices the change)
#   + 1/REFRESH_PER_SECOND (Rich Live paints it)
#
# Target: under 100ms.  Perceptible typing lag starts around 100ms.

_MAX_KEYSTROKE_LATENCY = 0.100  # seconds


def test_poll_interval_is_positive():
    assert POLL_INTERVAL > 0


def test_refresh_rate_is_positive():
    assert REFRESH_PER_SECOND > 0


def test_worst_case_keystroke_latency():
    worst_case = POLL_INTERVAL + 1 / REFRESH_PER_SECOND
    assert worst_case < _MAX_KEYSTROKE_LATENCY, (
        f"Worst-case latency {worst_case * 1000:.0f}ms exceeds "
        f"{_MAX_KEYSTROKE_LATENCY * 1000:.0f}ms target"
    )


def test_poll_interval_not_faster_than_refresh():
    """No point polling faster than Rich can repaint."""
    assert POLL_INTERVAL >= 1 / REFRESH_PER_SECOND - 0.001


# ── Other timing invariants ──────────────────────────────────────────


def test_completion_timeout_is_bounded():
    """Completion subprocess must not block longer than a few seconds."""
    assert 0 < SHELL_COMPLETION_TIMEOUT <= 5.0


def test_double_ctrl_c_window_is_reasonable():
    """Window must be short enough to feel responsive, long enough to be intentional."""
    assert 0.2 <= DOUBLE_CTRL_C_WINDOW <= 1.0
