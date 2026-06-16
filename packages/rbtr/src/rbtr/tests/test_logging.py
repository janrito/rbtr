"""Behaviour of `rbtr.logging`.

Log *events* (names, fields, levels, bound contextvars) are asserted on
captured event dicts via structlog's `LogCapture` — the autouse
`log_output` fixture, exercised by the daemon correlation tests.
Rendering itself is structlog's responsibility and is not re-tested
here.

These tests cover `configure_logging`'s own wiring — renderer
selection, root level, and the daemon's rotating file sink — plus the
pure helpers.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Generator
from logging.handlers import RotatingFileHandler

import pytest

from rbtr.config import LogFormat, config
from rbtr.logging import _resolve_stream_format, configure_logging, elapsed_ms


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Generator[None]:
    """Snapshot and restore the root logger around each test.

    `configure_logging` mutates the root logger's handlers and level;
    without this, that state leaks into sibling tests.
    """
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield
    for added in root.handlers:
        if added not in handlers:
            added.close()
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def tty_stream(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """A text stream that reports itself as a terminal."""
    buf = io.StringIO()
    monkeypatch.setattr(buf, "isatty", lambda: True)
    return buf


@pytest.fixture
def pipe_stream(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """A text stream that reports itself as non-interactive (piped)."""
    buf = io.StringIO()
    monkeypatch.setattr(buf, "isatty", lambda: False)
    return buf


@pytest.fixture
def debug_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the configured log level to DEBUG."""
    monkeypatch.setattr(config, "log_level", "DEBUG")


@pytest.mark.parametrize("fmt", [LogFormat.CONSOLE, LogFormat.JSON])
def test_explicit_format_ignores_tty(tty_stream: io.StringIO, fmt: LogFormat) -> None:
    assert _resolve_stream_format(fmt, tty_stream) is fmt


def test_auto_uses_console_on_a_tty(tty_stream: io.StringIO) -> None:
    assert _resolve_stream_format(LogFormat.AUTO, tty_stream) is LogFormat.CONSOLE


def test_auto_uses_json_when_piped(pipe_stream: io.StringIO) -> None:
    assert _resolve_stream_format(LogFormat.AUTO, pipe_stream) is LogFormat.JSON


@pytest.mark.usefixtures("debug_level")
def test_configure_sets_root_level() -> None:
    configure_logging(cache=False)
    assert logging.getLogger().level == logging.DEBUG


def test_cli_uses_a_stream_handler() -> None:
    configure_logging(cache=False)
    handlers = logging.getLogger().handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert not isinstance(handlers[0], logging.FileHandler)


def test_reconfigure_replaces_handlers() -> None:
    # Re-calling must not accumulate handlers (which would double every
    # log line); the root keeps exactly one.
    configure_logging(cache=False)
    configure_logging(cache=False)
    assert len(logging.getLogger().handlers) == 1


def test_daemon_uses_rotating_file_handler() -> None:
    configure_logging(to_file=True, cache=False)
    rotating = [h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)]
    assert len(rotating) == 1
    assert rotating[0].maxBytes == config.log_max_bytes
    assert rotating[0].backupCount == config.log_backup_count


def test_elapsed_ms_is_nonnegative_float() -> None:
    elapsed = elapsed_ms(time.perf_counter())
    assert isinstance(elapsed, float)
    assert elapsed >= 0
