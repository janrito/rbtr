"""What Ruby's call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `bare-super`: a bare `super` calls the base's method of the caller's own
  name.
- `no-parentheses`: a method called with no receiver, parentheses or
  arguments (`check`) is a call, being no local.
- `describe-block`: an RSpec `describe Greeter` block is a class chunk named
  `Greeter`, so a call to `Greeter` reaches it as well as the class.
- `member-order`: which inherited method a call reaches, by Ruby's
  ancestors: the class, then its modules from the last included, then its
  superclass; a module already among them is not added again.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "calls" / str(request.param)


@pytest.mark.parametrize("sample_dir", ["bare-super"], indirect=True)
def test_a_bare_super_reaches_the_bases_method_of_the_same_name(sample_edges: list[str]) -> None:
    """`super` in `Order#save` calls `Model#save`."""
    assert "order.rb::save -> order.rb::save [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["no-parentheses"], indirect=True)
def test_a_call_with_no_parentheses_reaches_the_method(sample_edges: list[str]) -> None:
    """`check` alone in `save` calls `Order#check`."""
    assert "order.rb::save -> order.rb::check [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["describe-block"], indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason="names only: a spec block named after a class is a class chunk of that "
    "name, and a call reaches every definition of its name",
)
def test_a_constructor_call_reaches_the_class_not_its_spec(sample_edges: list[str]) -> None:
    """`Greeter.new` reaches `class Greeter` alone, not `RSpec.describe Greeter`."""
    calls = [edge for edge in sample_edges if edge.startswith("greeter.rb::build ->")]
    assert calls == ["greeter.rb::build -> greeter.rb::Greeter [calls]"]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_override_hides_its_superclasss_method(sample_edges: list[str]) -> None:
    """`Special < Order < Base`, `Order#save` overriding: `self.save` is `Order#save`."""
    assert [e for e in sample_edges if e.startswith("override/special.rb::run ->")] == [
        "override/special.rb::run -> override/base.rb::check [calls]",
        "override/special.rb::run -> override/order.rb::save [calls]",
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_the_last_included_module_wins(sample_edges: list[str]) -> None:
    """`include A` then `include B`, both defining `m`: `self.m` is `B#m`."""
    assert [e for e in sample_edges if e.startswith("include/c.rb::run ->")] == [
        "include/c.rb::run -> include/b.rb::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_included_module_precedes_the_superclass(sample_edges: list[str]) -> None:
    """`C < Base` with `include Mixin`, both defining `m`: `self.m` is `Mixin#m`."""
    assert [e for e in sample_edges if e.startswith("superclass/c.rb::run ->")] == [
        "superclass/c.rb::run -> superclass/mixin.rb::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_later_module_precedes_an_earlier_modules_module(sample_edges: list[str]) -> None:
    """`include A` (which includes `X`) then `include B`, `X` and `B` defining `m`: ancestors C, B, A, X, so `B#m`."""
    assert [e for e in sample_edges if e.startswith("deeper/c.rb::run ->")] == [
        "deeper/c.rb::run -> deeper/b.rb::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_diamonds_shared_module_comes_last(sample_edges: list[str]) -> None:
    """`include Left` then `include Right`, both including `Top`, `Right#m` overriding: `self.m` is `Right#m`."""
    assert [e for e in sample_edges if e.startswith("diamond/bottom.rb::run ->")] == [
        "diamond/bottom.rb::run -> diamond/right.rb::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason="Ruby searches bases last written first, but one `include` listing "
    "several modules puts its first module first",
)
def test_one_include_listing_two_modules_puts_the_first_first(sample_edges: list[str]) -> None:
    """`include A, B`, both defining `m`: ancestors C, A, B, so `self.m` is `A#m`."""
    assert [e for e in sample_edges if e.startswith("multi/c.rb::run ->")] == [
        "multi/c.rb::run -> multi/a.rb::m [calls]"
    ]
