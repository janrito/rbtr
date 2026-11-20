"""What Go's call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `receiver`: a call through a method's receiver (`o.check()` in
  `func (o *Order)`) is a call on an `Order`, the receiver's type.
- `package-name`: an import binds the package's declared name, which may
  differ from its directory, the path's last segment.
- `member-order`: which promoted method a call on an embedding struct
  reaches: the shallowest embedding wins, whatever the field order, and a
  struct's own method hides a promoted one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        # `o.check()` in `func (o *Order) Save()` calls `Order.check`.
        pytest.param(
            "receiver",
            {
                ("order.go", "check"): {("order.go", "Save", "calls")},
            },
            id="a-call-through-the-receiver-reaches-the-types-method",
        ),
        # `import "example.com/m/strs"`, a directory holding `package util`, binds `util`.
        pytest.param(
            "package-name",
            {
                ("strs/trim.go", "Trim"): {
                    ("main.go", '"example.com/m/strs"', "imports"),
                    ("main.go", "run", "calls"),
                },
            },
            id="an-import-binds-the-packages-declared-name",
            marks=pytest.mark.xfail(
                strict=True,
                reason="an import binds the path's last segment; the package's declared name, "
                "which Go binds, is in the imported files, not the import",
            ),
        ),
        # `Special` embeds `Order` embeds `Base`, `Order.Save` hiding `Base.Save`: `s.Save()` is `Order.Save`.
        pytest.param(
            "member-order",
            {
                ("override/base.go", "Check"): {
                    ("override/base.go", "Save", "calls"),
                    ("override/special.go", "Run", "calls"),
                },
                ("override/base.go", "Save"): set(),
                ("override/order.go", "Save"): {("override/special.go", "Run", "calls")},
            },
            id="an-own-method-hides-a-promoted-one",
        ),
        # `C` embeds `A` then `B`, `A` embeds `X`, `X` and `B` defining `M`: `c.M()` is `B.M`, depth 1.
        pytest.param(
            "member-order",
            {
                ("deeper/b.go", "M"): {("deeper/c.go", "Run", "calls")},
                ("deeper/x.go", "M"): set(),
            },
            id="the-shallowest-promoted-method-wins",
        ),
        # `Bottom` embeds `Left` and `Right`, both embedding `Top`, `Right.M` hiding `Top.M`: `b.M()` is `Right.M`.
        pytest.param(
            "member-order",
            {
                ("diamond/right.go", "M"): {("diamond/bottom.go", "Run", "calls")},
                ("diamond/top.go", "M"): set(),
            },
            id="a-diamonds-shallower-method-wins",
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
