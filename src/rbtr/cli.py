"""Entry point for rbtr."""

import logging
import logging.handlers
import sys
from pathlib import Path

from rbtr.config import WORKSPACE_DIR, config
from rbtr.tui import run

LOG_PATH = WORKSPACE_DIR / "rbtr.log"


def _configure_logging() -> None:
    """Set up rotating file logging — the TUI owns the terminal so we can't use stderr.

    Only ``rbtr.*`` loggers use the configured level.  Third-party
    libraries stay at WARNING to avoid flooding the log with Rich
    renders, httpx requests, etc.

    Rotation: ``config.log.max_bytes`` per file, ``config.log.backup_count``
    backups (e.g. ``rbtr.log``, ``rbtr.log.1``, ``rbtr.log.2``).
    """
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    handler = logging.handlers.RotatingFileHandler(
        str(LOG_PATH),
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


def main() -> None:
    """Launch rbtr."""
    pr_number = _parse_args()
    _configure_logging()
    run(pr_number)


def _parse_args() -> int | None:
    """Extract an optional PR number from sys.argv."""
    if len(sys.argv) <= 1:
        return None
    arg = sys.argv[1]
    if arg in ("-h", "--help"):
        print("Usage: rbtr [pr_number]")
        print()
        print("  Launch rbtr in the current git repository.")
        print("  Optionally pass a PR number to review directly.")
        sys.exit(0)
    try:
        return int(arg)
    except ValueError:
        print(f"Unknown argument: {arg}")
        print("Usage: rbtr [pr_number]")
        sys.exit(1)
