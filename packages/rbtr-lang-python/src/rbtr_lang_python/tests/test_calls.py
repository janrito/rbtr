"""Python's call edges, each a small project under `samples/calls/`.

- `rebinding`: an import in a `try` body and one in its `except` body are
  alternatives, so a call reaches both; a later import of a name in the
  same block replaces the earlier, which names alone do not tell.
- `member-order`: which inherited member a call on the instance reaches,
  by Python's method resolution order (C3): an override hides what it
  overrides, the first listed base wins, a base's base precedes a later
  base, and a diamond's shared base comes last.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.domain.models import EdgeKind
from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        pytest.param(
            "rebinding",
            {
                ("fast.py", "loads"): {("app.py", "run", "calls")},
                ("slow.py", "loads"): {("app.py", "run", "calls")},
            },
            id="alternative-imports-are-both-reached",
        ),
        pytest.param(
            "rebinding",
            {
                ("two.py", "second"): {("app.py", "run", "calls")},
                ("one.py", "first"): set(),
            },
            id="a-later-import-in-one-block-replaces-the-earlier",
            marks=pytest.mark.xfail(
                strict=True,
                reason="names only: a later import binding the same name shadows the earlier, "
                "but resolution keeps no order and reaches both",
            ),
        ),
        pytest.param(
            "member-order",
            {
                ("override/order.py", "save"): {("override/special.py", "run", "calls")},
                ("override/base.py", "save"): {("override/order.py", "save", "calls")},
                ("override/base.py", "check"): {
                    ("override/base.py", "save", "calls"),
                    ("override/special.py", "run", "calls"),
                },
            },
            id="an-override-hides-its-bases-member",
        ),
        pytest.param(
            "member-order",
            {
                ("two/a.py", "m"): {("two/c.py", "run", "calls")},
                ("two/b.py", "m"): set(),
            },
            id="the-first-listed-base-wins",
        ),
        pytest.param(
            "member-order",
            {
                ("deeper/x.py", "m"): {("deeper/c.py", "run", "calls")},
                ("deeper/b.py", "m"): set(),
            },
            id="a-bases-base-precedes-a-later-base",
        ),
        pytest.param(
            "member-order",
            {
                ("diamond/right.py", "m"): {("diamond/bottom.py", "run", "calls")},
                ("diamond/top.py", "m"): set(),
            },
            id="a-diamonds-shared-base-comes-last",
        ),
    ],
)
def test_a_call_reaches_the_member_it_resolves_to(
    indexed: Indexed,
    sample: str,
    reached: dict[Target, set[Referrer]],
) -> None:
    """Each target `(path, symbol)` is called by exactly the callers named."""
    with indexed(Path(__file__).parent / "samples" / "calls" / sample) as index:
        assert_reached(index, reached, edges={EdgeKind.CALLS})
