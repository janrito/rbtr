"""How C's calls reach functions, each a small project under `samples/calls/`.

- `prototype`: a call sees the prototype its header declares, and through
  it the definition in another `.c` file, which no include brings in.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        # `helper()` in `main.c` calls `util.h`'s prototype and the `helper` `util.c` defines.
        pytest.param(
            "prototype",
            {
                ("util.c", "helper"): {("main.c", "run", "calls")},
                ("util.h", "helper"): {
                    ("main.c", '#include "util.h"', "imports"),
                    ("main.c", "run", "calls"),
                    ("util.c", '#include "util.h"', "imports"),
                },
            },
            id="a-call-reaches-the-prototype-and-the-definition",
        ),
    ],
)
def test_each_target_is_reached_by_exactly_the_chunks_named(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` of the sample under `samples/calls/` is reached by exactly the chunks named, by any edge."""
    with indexed(Path(__file__).parent / "samples" / "calls" / sample) as index:
        assert_reached(index, reached)
