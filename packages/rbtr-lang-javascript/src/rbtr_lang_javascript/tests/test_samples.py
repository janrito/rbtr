"""JS / TS / TSX sample extraction: each sample project, built as rbtr builds a repository and read as an agent reads it.

Parametrised over the package's three ids. The `javascript` sample includes a
`styles.css`, extracted via the css plugin (a dev dependency), so its edges
include the cross-file links.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_every_line_in_a_chunk, assert_reached
from rbtr.testing.reads import outline

if TYPE_CHECKING:
    from syrupy.assertion import SnapshotAssertion


@pytest.mark.parametrize("sample", ["javascript", "typescript", "tsx"])
def test_extraction_matches_snapshot(
    indexed: Indexed, sample: str, snapshot_json: SnapshotAssertion
) -> None:
    with indexed(Path(__file__).parent / "samples" / sample) as build:
        symbols = outline(build)
    assert symbols == snapshot_json


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        pytest.param(
            "javascript",
            {
                ("config.js", "LOCALE"): {
                    ("javascript.js", 'import { LOCALE } from "./config.js";', "imports")
                },
                ("javascript.js", "DEFAULT_GREETING"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "Greeter"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports"),
                    ("orders.js", "save", "calls"),
                },
                ("javascript.js", "cachedDefault"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "constructor"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "fallback"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "formatGreeting"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports"),
                    (
                        "orders.js",
                        'import { formatGreeting as greet } from "./javascript.js";',
                        "imports",
                    ),
                    ("orders.js", "save", "calls"),
                },
                ("javascript.js", "greet"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "helpers"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "locale, fallback"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "recipients"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("javascript.js", "shout"): {
                    ("orders.js", 'import * as greeter from "./javascript.js";', "imports")
                },
                ("orders.js", "Model"): {("orders.js", "Order", "inherits")},
                ("orders.js", "Order"): {("orders.js", "inner", "calls")},
                ("orders.js", "check"): {
                    ("orders.js", "save", "calls"),
                    ("orders.js", "total", "calls"),
                },
                ("orders.js", "inner"): {("orders.js", "build", "calls")},
                ("orders.js", "save"): {("orders.js", "save", "calls")},
                ("src/b.js", "helper"): {
                    ("src/a.js", "import { helper } from './b.js';", "imports")
                },
                ("styles.css", ".greeter"): {
                    ("javascript.js", 'import "./styles.css";', "imports")
                },
            },
            id="javascript",
        ),
        pytest.param(
            "typescript",
            {
                ("config.ts", "LOCALE"): {
                    ("typescript.ts", 'export * from "./config";', "imports"),
                    ("typescript.ts", 'import { LOCALE } from "./config";', "imports"),
                },
                ("orders.ts", "Model"): {("orders.ts", "Order", "inherits")},
                ("orders.ts", "check"): {("orders.ts", "save", "calls")},
                ("orders.ts", "save"): {("orders.ts", "save", "calls")},
                ("src/b.ts", "helper"): {("src/a.ts", "import { helper } from './b';", "imports")},
                ("types.ts", "Formatter"): {
                    ("typescript.ts", 'import type { Formatter } from "./types";', "imports")
                },
                ("typescript.ts", "AbstractGreeter"): {("typescript.ts", "Greeter", "inherits")},
            },
            id="typescript",
        ),
        pytest.param(
            "tsx",
            {
                ("labels.ts", "INITIAL_LABEL"): {
                    ("tsx.tsx", 'import { INITIAL_LABEL } from "./labels";', "imports")
                },
                ("src/b.tsx", "Helper"): {
                    ("src/a.tsx", "import { Helper } from './b';", "imports")
                },
            },
            id="tsx",
        ),
    ],
)
def test_every_edge_reaches_what_it_names(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target the sample's edges reach is reached by exactly the chunks named, by any edge."""
    with indexed(Path(__file__).parent / "samples" / sample) as build:
        assert_reached(build, reached)


@pytest.mark.parametrize("sample", ["javascript", "typescript", "tsx"])
def test_every_line_is_in_a_chunk(indexed: Indexed, sample: str) -> None:
    """No line of the sample's code falls outside a chunk."""
    with indexed(Path(__file__).parent / "samples" / sample) as build:
        assert_every_line_in_a_chunk(build)
