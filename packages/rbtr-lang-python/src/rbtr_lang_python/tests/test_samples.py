"""Python sample extraction: the `samples/python/` project, built as rbtr builds a repository and outlined with `list-symbols`.

The snapshots are the golden record of what Python extraction produces.
Engine-wide invariants are covered once in core.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rbtr.testing import Indexed
from rbtr.testing.assertions import assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


def test_extraction_matches_snapshot(indexed: Indexed, snapshot_json: SnapshotAssertion) -> None:
    with indexed(Path(__file__).parent / "samples" / "python") as build:
        symbols = outline(build)
    assert symbols == snapshot_json


def test_every_edge_reaches_what_it_names(indexed: Indexed) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    reached = {
        ("config.py", "LOCALE"): {
            ("orders.py", "from .config import LOCALE as locale_name", "imports"),
            ("python.py", "from .config import LOCALE", "imports"),
        },
        ("orders.py", "Model"): {("orders.py", "Order", "inherits")},
        ("orders.py", "Order"): {("orders.py", "inner", "calls")},
        ("orders.py", "check"): {("orders.py", "save", "calls"), ("orders.py", "total", "calls")},
        ("orders.py", "inner"): {("orders.py", "build", "calls")},
        ("orders.py", "register"): {("orders.py", "build", "calls")},
        ("orders.py", "save"): {("orders.py", "save", "calls")},
        ("pkg/b.py", "helper"): {("pkg/a.py", "from pkg.b import helper", "imports")},
        ("python.py", "DEFAULT_GREETING"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "Greeter"): {
            ("orders.py", "from .python import *", "imports"),
            ("orders.py", "save", "calls"),
        },
        ("python.py", "GreetingList"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "LOCALES, FALLBACK"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "MAX_RECIPIENTS"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "__init__"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "cached_default"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "config_path"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "default"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "fetch_remote_greeting"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "format_greeting"): {
            ("orders.py", "", "calls"),
            ("orders.py", "from .python import *", "imports"),
            ("python.py", "greet", "calls"),
        },
        ("python.py", "greet"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "normalise"): {
            ("orders.py", "from .python import *", "imports"),
            ("python.py", "format_greeting", "calls"),
        },
        ("python.py", "seen"): {("orders.py", "from .python import *", "imports")},
        ("python.py", "shout"): {("orders.py", "from .python import *", "imports")},
    }
    with indexed(Path(__file__).parent / "samples" / "python") as build:
        assert_reached(build, reached)


def test_every_line_is_in_a_chunk(indexed: Indexed) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / "python") as build:
        assert_every_line_in_a_chunk(build)
