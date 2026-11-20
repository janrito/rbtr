"""What Ruby's call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `bare-super`: a bare `super` calls the base's method of the caller's own
  name.
- `no-parentheses`: a method called with no receiver, parentheses or
  arguments (`check`) is a call, being no local.
- `describe-block`: an RSpec `describe Greeter` block is a test, which a
  call to `Greeter` does not reach.
- `member-order`: which inherited method a call reaches, by Ruby's
  ancestors: the class, then its modules from the last included, then its
  superclass; a module already among them is not added again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rbtr.testing import Indexed
from rbtr.testing.assertions import Referrer, Target, assert_reached


@pytest.mark.parametrize(
    ("sample", "reached"),
    [
        # `super` in `Order#save` calls `Model#save`.
        pytest.param(
            "bare-super",
            {
                ("order.rb", "save"): {("order.rb", "save", "calls")},
            },
            id="a-bare-super-reaches-the-bases-method-of-the-same-name",
        ),
        # `check` alone in `save` calls `Order#check`.
        pytest.param(
            "no-parentheses",
            {
                ("order.rb", "check"): {("order.rb", "save", "calls")},
            },
            id="a-call-with-no-parentheses-reaches-the-method",
        ),
        # `Greeter.new` reaches `class Greeter` alone, not `RSpec.describe Greeter`.
        pytest.param(
            "describe-block",
            {
                ("greeter.rb", "Greeter"): {("greeter.rb", "build", "calls")},
            },
            id="a-constructor-call-reaches-the-class-not-its-spec",
        ),
        # `Special < Order < Base`, `Order#save` overriding: `self.save` is `Order#save`.
        pytest.param(
            "member-order",
            {
                ("override/base.rb", "check"): {
                    ("override/base.rb", "save", "calls"),
                    ("override/order.rb", 'require_relative "base"', "imports"),
                    ("override/special.rb", "run", "calls"),
                },
                ("override/base.rb", "save"): {
                    ("override/order.rb", 'require_relative "base"', "imports"),
                    ("override/order.rb", "save", "calls"),
                },
                ("override/order.rb", "save"): {
                    ("override/special.rb", 'require_relative "order"', "imports"),
                    ("override/special.rb", "run", "calls"),
                },
            },
            id="an-override-hides-its-superclasss-method",
        ),
        # `include A` then `include B`, both defining `m`: `self.m` is `B#m`.
        pytest.param(
            "member-order",
            {
                ("include/a.rb", "m"): {("include/c.rb", 'require_relative "a"', "imports")},
                ("include/b.rb", "m"): {
                    ("include/c.rb", 'require_relative "b"', "imports"),
                    ("include/c.rb", "run", "calls"),
                },
            },
            id="the-last-included-module-wins",
        ),
        # `C < Base` with `include Mixin`, both defining `m`: `self.m` is `Mixin#m`.
        pytest.param(
            "member-order",
            {
                ("superclass/base.rb", "m"): {
                    ("superclass/c.rb", 'require_relative "base"', "imports")
                },
                ("superclass/mixin.rb", "m"): {
                    ("superclass/c.rb", 'require_relative "mixin"', "imports"),
                    ("superclass/c.rb", "run", "calls"),
                },
            },
            id="an-included-module-precedes-the-superclass",
        ),
        # `include A` (which includes `X`) then `include B`, `X` and `B` defining `m`: ancestors C, B, A, X, so `B#m`.
        pytest.param(
            "member-order",
            {
                ("deeper/b.rb", "m"): {
                    ("deeper/c.rb", 'require_relative "b"', "imports"),
                    ("deeper/c.rb", "run", "calls"),
                },
                ("deeper/x.rb", "m"): {("deeper/a.rb", 'require_relative "x"', "imports")},
            },
            id="a-later-module-precedes-an-earlier-modules-module",
        ),
        # `include Left` then `include Right`, both including `Top`, `Right#m` overriding: `self.m` is `Right#m`.
        pytest.param(
            "member-order",
            {
                ("diamond/right.rb", "m"): {
                    ("diamond/bottom.rb", 'require_relative "right"', "imports"),
                    ("diamond/bottom.rb", "run", "calls"),
                },
                ("diamond/top.rb", "m"): {
                    ("diamond/left.rb", 'require_relative "top"', "imports"),
                    ("diamond/right.rb", 'require_relative "top"', "imports"),
                },
            },
            id="a-diamonds-shared-module-comes-last",
        ),
        # `include A, B`, both defining `m`: ancestors C, A, B, so `self.m` is `A#m`.
        pytest.param(
            "member-order",
            {
                ("multi/a.rb", "m"): {
                    ("multi/c.rb", 'require_relative "a"', "imports"),
                    ("multi/c.rb", "run", "calls"),
                },
                ("multi/b.rb", "m"): {("multi/c.rb", 'require_relative "b"', "imports")},
            },
            id="one-include-listing-two-modules-puts-the-first-first",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Ruby searches bases last written first, but one `include` listing "
                "several modules puts its first module first",
            ),
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
