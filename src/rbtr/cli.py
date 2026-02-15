"""Entry point for rbtr."""

import sys

from rbtr.tui import run


def main() -> None:
    """Launch rbtr."""
    pr_number = _parse_args()
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
