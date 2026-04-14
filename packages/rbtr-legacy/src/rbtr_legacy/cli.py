"""Entry point for rbtr."""

import sys
from dataclasses import dataclass

from rbtr_legacy import log
from rbtr_legacy.tui.ui import run


@dataclass(frozen=True, slots=True)
class StartupOpts:
    """Parsed command-line options."""

    pr_number: int | None = None
    snapshot_ref: str | None = None
    continue_session: bool = False


def main() -> None:
    """Launch rbtr."""
    opts = _parse_args()
    log.configure()
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
