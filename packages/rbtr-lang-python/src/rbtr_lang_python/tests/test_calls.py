"""Python's call edges, each a small project under `samples/calls/`.

- `rebinding`: an import in a `try` body and one in its `except` body are
  alternatives, so a call reaches both; a later import of a name in the
  same block replaces the earlier, which names alone do not tell.
- `member-order`: which inherited member a call on the instance reaches,
  by Python's method resolution order (C3): a base's base precedes a later
  base, and an override hides what it overrides.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "calls" / str(request.param)


@pytest.mark.parametrize("sample_dir", ["rebinding"], indirect=True)
def test_alternative_imports_are_both_reached(sample_edges: list[str]) -> None:
    calls = [edge for edge in sample_edges if "[calls]" in edge]
    assert "app.py::run -> fast.py::loads [calls]" in calls
    assert "app.py::run -> slow.py::loads [calls]" in calls


@pytest.mark.parametrize("sample_dir", ["rebinding"], indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason="names only: a later import binding the same name shadows the earlier, "
    "but resolution keeps no order and reaches both",
)
def test_a_later_import_in_one_block_replaces_the_earlier(sample_edges: list[str]) -> None:
    calls = [edge for edge in sample_edges if "[calls]" in edge]
    assert "app.py::run -> two.py::second [calls]" in calls
    assert "app.py::run -> one.py::first [calls]" not in calls


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_override_hides_its_bases_member(sample_edges: list[str]) -> None:
    """`Special(Order(Base))`, `Order.save` overriding `Base.save`: `self.save()` is `Order.save`."""
    assert [e for e in sample_edges if e.startswith("override/special.py::run ->")] == [
        "override/special.py::run -> override/base.py::check [calls]",
        "override/special.py::run -> override/order.py::save [calls]",
    ]
    assert "override/order.py::save -> override/base.py::save [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_the_first_listed_base_wins(sample_edges: list[str]) -> None:
    """`class C(A, B)`, both defining `m`: `self.m()` is `A.m`."""
    assert [e for e in sample_edges if e.startswith("two/c.py::run ->")] == [
        "two/c.py::run -> two/a.py::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_bases_base_precedes_a_later_base(sample_edges: list[str]) -> None:
    """`class C(A, B)`, `class A(X)`, `X` and `B` defining `m`: the order is C, A, X, B, so `X.m`."""
    assert [e for e in sample_edges if e.startswith("deeper/c.py::run ->")] == [
        "deeper/c.py::run -> deeper/x.py::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_diamonds_shared_base_comes_last(sample_edges: list[str]) -> None:
    """`Bottom(Left, Right)`, both from `Top`, `Right.m` overriding `Top.m`: `self.m()` is `Right.m`."""
    assert [e for e in sample_edges if e.startswith("diamond/bottom.py::run ->")] == [
        "diamond/bottom.py::run -> diamond/right.py::m [calls]"
    ]
