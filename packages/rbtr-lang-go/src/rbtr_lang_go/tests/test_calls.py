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


@pytest.fixture(scope="module")
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "calls" / str(request.param)


@pytest.mark.parametrize("sample_dir", ["receiver"], indirect=True)
def test_a_call_through_the_receiver_reaches_the_types_method(sample_edges: list[str]) -> None:
    """`o.check()` in `func (o *Order) Save()` calls `Order.check`."""
    assert "order.go::Save -> order.go::check [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["package-name"], indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason="an import binds the path's last segment; the package's declared name, "
    "which Go binds, is in the imported files, not the import",
)
def test_an_import_binds_the_packages_declared_name(sample_edges: list[str]) -> None:
    """`import "example.com/m/strs"`, a directory holding `package util`, binds `util`."""
    assert "main.go::run -> strs/trim.go::Trim [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_own_method_hides_a_promoted_one(sample_edges: list[str]) -> None:
    """`Special` embeds `Order` embeds `Base`, `Order.Save` hiding `Base.Save`: `s.Save()` is `Order.Save`."""
    assert [e for e in sample_edges if e.startswith("override/special.go::Run ->")] == [
        "override/special.go::Run -> override/base.go::Check [calls]",
        "override/special.go::Run -> override/order.go::Save [calls]",
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_the_shallowest_promoted_method_wins(sample_edges: list[str]) -> None:
    """`C` embeds `A` then `B`, `A` embeds `X`, `X` and `B` defining `M`: `c.M()` is `B.M`, depth 1."""
    assert [e for e in sample_edges if e.startswith("deeper/c.go::Run ->")] == [
        "deeper/c.go::Run -> deeper/b.go::M [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_diamonds_shallower_method_wins(sample_edges: list[str]) -> None:
    """`Bottom` embeds `Left` and `Right`, both embedding `Top`, `Right.M` hiding `Top.M`: `b.M()` is `Right.M`."""
    assert [e for e in sample_edges if e.startswith("diamond/bottom.go::Run ->")] == [
        "diamond/bottom.go::Run -> diamond/right.go::M [calls]"
    ]
