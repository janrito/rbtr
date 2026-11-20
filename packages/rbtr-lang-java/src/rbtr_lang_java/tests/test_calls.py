"""What Java's call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `static-import`: a static import binds a class's member, listed by its
  path in the class (`Strings.shout`).
- `member-order`: which inherited member a call reaches: a class's own
  chain before any interface's default method, and of interfaces the most
  specific (one extending another overrides it).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        # `import static util.Strings.shout` then `shout("hi")` calls `Strings.shout`.
        pytest.param(
            "static-import",
            {
                ("util/Strings.java", "shout"): {("App.java", "run", "calls")},
            },
            id="a-static-import-reaches-the-member-it-names",
        ),
        # `Special extends Order extends Base`, `Order.save` overriding: `save()` is `Order.save`.
        pytest.param(
            "member-order",
            {
                ("override/Base.java", "check"): {
                    ("override/Base.java", "save", "calls"),
                    ("override/Special.java", "run", "calls"),
                },
                ("override/Base.java", "save"): {("override/Order.java", "save", "calls")},
                ("override/Order.java", "save"): {("override/Special.java", "run", "calls")},
            },
            id="an-override-hides-its-bases-member",
        ),
        # `C extends B implements I`, both defining `m`: `m()` is `B.m`.
        pytest.param(
            "member-order",
            {
                ("classwins/B.java", "m"): {("classwins/C.java", "run", "calls")},
                ("classwins/I.java", "m"): set(),
            },
            id="a-classs-method-wins-over-an-interfaces-default",
        ),
        # `K implements I, J`, `J extends I`, both defining `m`: `m()` is `J.m`.
        pytest.param(
            "member-order",
            {
                ("specific/I.java", "m"): set(),
                ("specific/J.java", "m"): {("specific/K.java", "run", "calls")},
            },
            id="the-most-specific-interface-default-wins",
        ),
        # `Bottom implements Left, Right`, both extending `Top`, `Right.m` overriding: `m()` is `Right.m`.
        pytest.param(
            "member-order",
            {
                ("diamond/Right.java", "m"): {("diamond/Bottom.java", "run", "calls")},
                ("diamond/Top.java", "m"): set(),
            },
            id="a-diamonds-overriding-interface-wins",
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
