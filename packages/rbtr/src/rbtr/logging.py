"""Central logging configuration for rbtr.

All of rbtr's own code logs through `structlog.get_logger(__name__)`,
producing event dictionaries. A single stdlib handler renders both
structlog events and third-party `logging` records (llama.cpp,
huggingface_hub, duckdb, pyzmq, pygit2) through one
`structlog.stdlib.ProcessorFormatter`, so every line shares the same
fields and format.

Three sinks, selected by caller and `config.log_format`:

- **CLI** (`configure_logging()`)        — a `StreamHandler` on
  `sys.stderr`.  Console renderer on a TTY, JSON otherwise.  stdout
  is reserved for command output (`cli.output.emit`); logs never go
  there.
- **Daemon** (`configure_logging(to_file=True)`) — a
  `RotatingFileHandler` on `config.daemon_log`, always JSON lines,
  sized by `config.log_max_bytes` / `config.log_backup_count`.
- **Third-party**                        — stdlib records propagate to
  the root logger and render through the same formatter via
  `foreign_pre_chain`.

`configure_logging` is idempotent: it replaces the root logger's
handlers on every call.  Pass `cache=False` in tests so per-test
reconfiguration takes effect (structlog caches bound loggers on first
use otherwise).

The correlation keys (`request_id`, `job_id`) are bound at runtime via
`structlog.contextvars`; the pipeline merges them automatically.  See
ARCHITECTURE.md's Observability section for the cross-module flow.
"""

from __future__ import annotations

import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import IO

import structlog

from rbtr.config import LogFormat, config


def elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a `time.perf_counter()` reading.

    A small helper for the `elapsed_ms` log field; pair it with a
    `start = time.perf_counter()` taken at the point being timed.
    """
    return round((time.perf_counter() - start) * 1000, 1)


# Processors applied to records that do NOT originate from structlog
# (third-party stdlib loggers), so they gain the same fields as
# structlog events.  Mirrors the structlog chain below.
_TIMESTAMPER = structlog.processors.TimeStamper(fmt="iso", utc=True)
_CALLSITE = structlog.processors.CallsiteParameterAdder(
    {
        structlog.processors.CallsiteParameter.FILENAME,
        structlog.processors.CallsiteParameter.FUNC_NAME,
        structlog.processors.CallsiteParameter.LINENO,
    }
)

_FOREIGN_PRE_CHAIN: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.ExtraAdder(),
    _TIMESTAMPER,
    _CALLSITE,
]

# Processors run on the structlog side before handing off to the
# stdlib `ProcessorFormatter`.
_STRUCTLOG_PROCESSORS: list[structlog.typing.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.PositionalArgumentsFormatter(),
    _TIMESTAMPER,
    structlog.processors.StackInfoRenderer(),
    _CALLSITE,
    structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
]


def _console_formatter() -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_FOREIGN_PRE_CHAIN,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
    )


def _json_formatter() -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_FOREIGN_PRE_CHAIN,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
    )


def _resolve_stream_format(fmt: LogFormat, stream: IO[str]) -> LogFormat:
    """Resolve `AUTO` for a stream: console on a TTY, else JSON."""
    if fmt is not LogFormat.AUTO:
        return fmt
    return LogFormat.CONSOLE if stream.isatty() else LogFormat.JSON


def configure_logging(*, to_file: bool = False, cache: bool = True) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent — replaces the root logger's handlers on each call.

    Args:
        to_file: When True (the daemon), write JSON lines to
            `config.daemon_log` via a rotating file handler.  When
            False (the CLI), write to `sys.stderr` using the renderer
            chosen by `config.log_format`.
        cache: Passed to structlog's `cache_logger_on_first_use`.
            Leave True in production; set False in tests so per-test
            reconfiguration is honoured.
    """
    structlog.configure(
        processors=_STRUCTLOG_PROCESSORS,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=cache,
    )

    handler: logging.Handler
    if to_file:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            config.daemon_log,
            maxBytes=config.log_max_bytes,
            backupCount=config.log_backup_count,
        )
        handler.setFormatter(_json_formatter())
    else:
        handler = logging.StreamHandler(sys.stderr)
        fmt = _resolve_stream_format(config.log_format, sys.stderr)
        handler.setFormatter(_json_formatter() if fmt is LogFormat.JSON else _console_formatter())

    root = logging.getLogger()
    for old in root.handlers[:]:
        old.close()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(config.log_level.upper())
