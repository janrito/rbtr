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


@pytest.fixture(scope="module")
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "calls" / str(request.param)


@pytest.mark.parametrize("sample_dir", ["out-of-class"], indirect=True)
def test_an_out_of_class_member_calls_through_its_class(sample_edges: list[str]) -> None:
    """`check()` in `Order::save`, defined in `save.cpp`, reaches `Order`'s `check` through its class."""
    assert "save.cpp::save -> order.hpp::check [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["out-of-class"], indirect=True)
def test_a_member_call_reaches_the_members_definition(sample_edges: list[str]) -> None:
    """`check()` in `Order::save` calls the `Order::check` `check.cpp` defines."""
    assert "save.cpp::save -> check.cpp::check [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["out-of-class"], indirect=True)
def test_a_member_defined_by_its_full_name_calls_through_its_class(sample_edges: list[str]) -> None:
    """`check()` in `void shop::Order::close()` reaches `Order`'s `check`."""
    assert "save.cpp::close -> order.hpp::check [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["c-header"], indirect=True)
def test_a_call_reaches_a_c_prototype_and_its_cpp_definition(sample_edges: list[str]) -> None:
    """`helper()` in `main.cpp` calls `util.h`'s prototype, read as C, and the `helper` `util.cpp` defines."""
    assert [e for e in sample_edges if e.startswith("main.cpp::run ->")] == [
        "main.cpp::run -> util.cpp::helper [calls]",
        "main.cpp::run -> util.h::helper [calls]",
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_override_hides_its_bases_member(sample_edges: list[str]) -> None:
    """`Special : Order : Base`, `Order::save` hiding `Base::save`: `save()` is `Order::save`."""
    assert [e for e in sample_edges if e.startswith("override/special.hpp::run ->")] == [
        "override/special.hpp::run -> override/base.hpp::check [calls]",
        "override/special.hpp::run -> override/order.hpp::save [calls]",
    ]
    assert "override/order.hpp::save -> override/base.hpp::save [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_override_dominates_through_a_virtual_base(sample_edges: list[str]) -> None:
    """`Bottom : Left, Right`, both virtually from `Top`, `Right::m` overriding: `m()` is `Right::m`."""
    assert [e for e in sample_edges if e.startswith("diamond/bottom.hpp::run ->")] == [
        "diamond/bottom.hpp::run -> diamond/right.hpp::m [calls]"
    ]
