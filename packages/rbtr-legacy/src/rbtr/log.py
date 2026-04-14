"""Logging configuration — the TUI owns the terminal, so all logs go to a file.

`configure()` is called once at startup from `cli.main()`.
`remove_stream_handlers()` is called after deferred imports of
libraries that install their own `StreamHandler` at init time.
"""

import logging
import logging.handlers

from rbtr.config import config
from rbtr.workspace import workspace_dir


def configure() -> None:
    """Set up rotating file logging.

    Only `rbtr.*` loggers use the configured level.  Third-party
    libraries stay at WARNING to avoid flooding the log with Rich
    renders, httpx requests, etc.

    Rotation: `config.log.max_bytes` per file, `config.log.backup_count`
    backups (e.g. `rbtr.log`, `rbtr.log.1`, `rbtr.log.2`).
    """
    log_path = workspace_dir() / "rbtr.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=config.log.max_bytes,
        backupCount=config.log.backup_count,
    )
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    rbtr_level = getattr(logging, config.log.level.upper(), logging.INFO)

    rbtr_logger = logging.getLogger("rbtr")
    rbtr_logger.setLevel(rbtr_level)
    rbtr_logger.addHandler(handler)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.addHandler(handler)

    # Redirect Python's last-resort handler to the log file so
    # loggers that fire before any handler is configured still
    # land in rbtr.log instead of corrupting the display.
    logging.lastResort = handler


def remove_stream_handlers(name: str) -> None:
    """Remove bare `StreamHandler`s that a library added to its own logger.

    Some libraries (e.g. `huggingface_hub`) unconditionally install
    a `StreamHandler` at import time.  Call this after the deferred
    import so log records propagate to the root file handler instead.

    Matches only exact `StreamHandler` instances (not subclasses
    like `FileHandler` or `RotatingFileHandler`).
    """
    logger = logging.getLogger(name)
    for h in logger.handlers:
        if type(h) is logging.StreamHandler:
            logger.removeHandler(h)
