"""What C++'s call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `out-of-class`: members declared in a header and defined in two `.cpp`
  files, one calling the other's.
- `c-header`: a C header's prototype, defined in a `.cpp` file.
- `member-order`: which inherited member a call reaches. A name found in
  two bases is ambiguous, a compile error, unless one hides the other: an
  override hides what it overrides, through a shared virtual base too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        # `check()` in `Order::save`, defined in `save.cpp`, reaches `Order`'s `check` through its class.
        pytest.param(
            "out-of-class",
            {
                ("order.hpp", "check"): {
                    ("check.cpp", '#include "order.hpp"', "imports"),
                    ("save.cpp", '#include "order.hpp"', "imports"),
                    ("save.cpp", "close", "calls"),
                    ("save.cpp", "save", "calls"),
                },
            },
            id="an-out-of-class-member-calls-through-its-class",
        ),
        # `check()` in `Order::save` calls the `Order::check` `check.cpp` defines.
        pytest.param(
            "out-of-class",
            {
                ("check.cpp", "check"): {
                    ("save.cpp", "close", "calls"),
                    ("save.cpp", "save", "calls"),
                },
            },
            id="a-member-call-reaches-the-members-definition",
        ),
        # `check()` in `void shop::Order::close()` reaches `Order`'s `check`.
        pytest.param(
            "out-of-class",
            {
                ("order.hpp", "check"): {
                    ("check.cpp", '#include "order.hpp"', "imports"),
                    ("save.cpp", '#include "order.hpp"', "imports"),
                    ("save.cpp", "close", "calls"),
                    ("save.cpp", "save", "calls"),
                },
            },
            id="a-member-defined-by-its-full-name-calls-through-its-class",
        ),
        # `helper()` in `main.cpp` calls `util.h`'s prototype, read as C, and the `helper` `util.cpp` defines.
        pytest.param(
            "c-header",
            {
                ("util.cpp", "helper"): {("main.cpp", "run", "calls")},
                ("util.h", "helper"): {
                    ("main.cpp", '#include "util.h"', "imports"),
                    ("main.cpp", "run", "calls"),
                    ("util.cpp", '#include "util.h"', "imports"),
                },
            },
            id="a-call-reaches-a-c-prototype-and-its-cpp-definition",
        ),
        # `Special : Order : Base`, `Order::save` hiding `Base::save`: `save()` is `Order::save`.
        pytest.param(
            "member-order",
            {
                ("override/base.hpp", "check"): {
                    ("override/base.hpp", "save", "calls"),
                    ("override/order.hpp", '#include "base.hpp"', "imports"),
                    ("override/special.hpp", "run", "calls"),
                },
                ("override/base.hpp", "save"): {
                    ("override/order.hpp", '#include "base.hpp"', "imports"),
                    ("override/order.hpp", "save", "calls"),
                },
                ("override/order.hpp", "save"): {
                    ("override/special.hpp", '#include "order.hpp"', "imports"),
                    ("override/special.hpp", "run", "calls"),
                },
            },
            id="an-override-hides-its-bases-member",
        ),
        # `Bottom : Left, Right`, both virtually from `Top`, `Right::m` overriding: `m()` is `Right::m`.
        pytest.param(
            "member-order",
            {
                ("diamond/right.hpp", "m"): {
                    ("diamond/bottom.hpp", '#include "right.hpp"', "imports"),
                    ("diamond/bottom.hpp", "run", "calls"),
                },
                ("diamond/top.hpp", "m"): {
                    ("diamond/left.hpp", '#include "top.hpp"', "imports"),
                    ("diamond/right.hpp", '#include "top.hpp"', "imports"),
                },
            },
            id="an-override-dominates-through-a-virtual-base",
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
