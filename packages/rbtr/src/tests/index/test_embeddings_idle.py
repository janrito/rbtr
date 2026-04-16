"""Tests for embedding idle unload."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest


class TestIdleMonitor:
    def test_start_with_zero_timeout_does_nothing(self) -> None:
        from rbtr.index import embeddings

        embeddings._idle_stop_event.set() if embeddings._idle_stop_event else None
        embeddings._idle_monitor_running = False

        embeddings.start_idle_monitor(0)

        assert not embeddings._idle_monitor_running

    def test_start_twice_does_nothing(self) -> None:
        from rbtr.index import embeddings

        # Reset state
        if embeddings._idle_stop_event:
            embeddings._idle_stop_event.set()
        embeddings._idle_monitor_running = False
        embeddings._idle_stop_event = None

        embeddings.start_idle_monitor(1)
        first_running = embeddings._idle_monitor_running

        embeddings.start_idle_monitor(1)  # second call

        assert first_running
        assert embeddings._idle_monitor_running  # still running, not restarted

        embeddings.stop_idle_monitor()

    def test_stop_idle_monitor_clears_flag(self) -> None:
        from rbtr.index import embeddings

        if embeddings._idle_stop_event:
            embeddings._idle_stop_event.set()
        embeddings._idle_monitor_running = False
        embeddings._idle_stop_event = None

        # stop on not-running is a no-op
        embeddings.stop_idle_monitor()
        assert not embeddings._idle_monitor_running

    def test_reset_model_clears_last_use_time(self) -> None:
        from rbtr.index import embeddings

        embeddings._last_use_time = 9999.0
        embeddings._model_ref = None

        embeddings.reset_model()

        assert embeddings._last_use_time == 0.0
