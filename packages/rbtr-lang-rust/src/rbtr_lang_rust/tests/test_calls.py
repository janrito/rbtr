"""What Rust's call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `crate-path`: a call written with its full path (`crate::util::log()`)
  starts at `crate`, which no `use` binds.
- `mod-declaration`: `mod util;` binds `util` to the module beside the
  file.
- `member-order`: which method a call on `self` reaches: an inherent
  method before a trait's of the same name, a trait's default method when
  the type has none, and a supertrait's from its subtrait.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        # `crate::util::log()` calls `log` in `src/util.rs`.
        pytest.param(
            "crate-path",
            {
                ("src/util.rs", "log"): {
                    ("src/main.rs", "main", "calls"),
                    ("src/main.rs", "mod util;", "imports"),
                },
            },
            id="a-call-by-its-full-path-reaches-the-function",
            marks=pytest.mark.xfail(
                strict=True,
                reason="a path's head is looked up among the names a file binds, and `crate`, "
                "`super` and modules declared with `mod` are bound by no import",
            ),
        ),
        # `Order` has `save` and implements `Saver` (`save`, `describe`): `self.save()` is `Order::save`, `self.describe()` `Saver::describe`.
        pytest.param(
            "member-order",
            {
                ("src/order.rs", "save"): {
                    ("src/main.rs", "mod order;", "imports"),
                    ("src/order.rs", "run", "calls"),
                },
                ("src/saver.rs", "describe"): {
                    ("src/main.rs", "mod saver;", "imports"),
                    ("src/order.rs", "run", "calls"),
                },
                ("src/saver.rs", "save"): {("src/main.rs", "mod saver;", "imports")},
            },
            id="an-inherent-method-precedes-a-traits",
        ),
        # `trait Titled: Named`: `self.name()` in `Titled::title` is `Named::name`.
        pytest.param(
            "member-order",
            {
                ("src/named.rs", "name"): {
                    ("src/main.rs", "mod named;", "imports"),
                    ("src/titled.rs", "title", "calls"),
                },
            },
            id="a-subtrait-reaches-its-supertraits-method",
        ),
        # `mod util;` binds `util` to `util.rs`: `util::log()` in `main` reaches `log`.
        pytest.param(
            "mod-declaration",
            {
                ("src/util.rs", "log"): {
                    ("src/main.rs", "main", "calls"),
                    ("src/main.rs", "mod util;", "imports"),
                },
            },
            id="a-declared-module-is-bound-by-its-name",
        ),
    ],
)
def test_each_target_is_reached_by_exactly_the_chunks_named(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` of the sample under `samples/calls/` is reached by exactly the chunks named, by any edge."""
    with indexed(Path(__file__).parent / "samples" / "calls" / sample) as build:
        assert_reached(build, reached)
