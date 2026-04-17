"""Entry point for ``python -m rbtr``.

Delegates to `rbtr.cli.main`, making the package runnable via
the same interpreter that imports it. This is the preferred
invocation for subprocess launches (e.g. spawning the daemon)
because it guarantees the spawned process uses the exact same
install \u2014 no PATH resolution, no version mismatches, no shims.
"""

from __future__ import annotations

from rbtr.cli import main

if __name__ == "__main__":
    main()
