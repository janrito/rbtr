"""Entry point for rbtr."""

import logging
import logging.handlers
import sys
from dataclasses import dataclass
from pathlib import Path

from rbtr.config import WORKSPACE_DIR, config
from rbtr.tui.ui import run

LOG_PATH = WORKSPACE_DIR / "rbtr.log"


@dataclass(frozen=True, slots=True)
class StartupOpts:
    """Parsed command-line options."""

    pr_number: int | None = None
    snapshot_ref: str | None = None
    continue_session: bool = False


def _configure_logging() -> None:
    """Set up rotating file logging — the TUI owns the terminal so we can't use stderr.

    Only `rbtr.*` loggers use the configured level.  Third-party
    libraries stay at WARNING to avoid flooding the log with Rich
    renders, httpx requests, etc.

    Rotation: `config.log.max_bytes` per file, `config.log.backup_count`
    backups (e.g. `rbtr.log`, `rbtr.log.1`, `rbtr.log.2`).
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
    opts = _parse_args()
    _configure_logging()
    run(
        pr_number=opts.pr_number,
        snapshot_ref=opts.snapshot_ref,
        continue_session=opts.continue_session,
    )


def _parse_args() -> StartupOpts:
    """Parse command-line arguments."""
    args = sys.argv[1:]

    if "-h" in args or "--help" in args:
        print("Usage: rbtr [-c | --continue] [<pr_number> | <ref>]")
        print()
        print("  Launch rbtr in the current git repository.")
        print()
        print("Options:")
        print("  -c, --continue   Resume the last session for this repo")
        print("  pr_number        Start a review for the given PR number")
        print("  ref              Snapshot review at a git ref (branch, tag, SHA, HEAD)")
        sys.exit(0)

    continue_session = "-c" in args or "--continue" in args
    if continue_session:
        args = [a for a in args if a not in ("-c", "--continue")]

    if not args:
        return StartupOpts(continue_session=continue_session)

    if continue_session:
        print("Cannot combine -c with a review target.")
        sys.exit(1)

    try:
        return StartupOpts(pr_number=int(args[0]))
    except ValueError:
        return StartupOpts(snapshot_ref=args[0])
