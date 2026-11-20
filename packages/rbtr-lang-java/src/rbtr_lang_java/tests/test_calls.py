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


@pytest.fixture(scope="module")
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "calls" / str(request.param)


@pytest.mark.parametrize("sample_dir", ["static-import"], indirect=True)
def test_a_static_import_reaches_the_member_it_names(sample_edges: list[str]) -> None:
    """`import static util.Strings.shout` then `shout("hi")` calls `Strings.shout`."""
    assert "App.java::run -> util/Strings.java::shout [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_override_hides_its_bases_member(sample_edges: list[str]) -> None:
    """`Special extends Order extends Base`, `Order.save` overriding: `save()` is `Order.save`."""
    assert [e for e in sample_edges if e.startswith("override/Special.java::run ->")] == [
        "override/Special.java::run -> override/Base.java::check [calls]",
        "override/Special.java::run -> override/Order.java::save [calls]",
    ]
    assert "override/Order.java::save -> override/Base.java::save [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_classs_method_wins_over_an_interfaces_default(sample_edges: list[str]) -> None:
    """`C extends B implements I`, both defining `m`: `m()` is `B.m`."""
    assert [e for e in sample_edges if e.startswith("classwins/C.java::run ->")] == [
        "classwins/C.java::run -> classwins/B.java::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_the_most_specific_interface_default_wins(sample_edges: list[str]) -> None:
    """`K implements I, J`, `J extends I`, both defining `m`: `m()` is `J.m`."""
    assert [e for e in sample_edges if e.startswith("specific/K.java::run ->")] == [
        "specific/K.java::run -> specific/J.java::m [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_diamonds_overriding_interface_wins(sample_edges: list[str]) -> None:
    """`Bottom implements Left, Right`, both extending `Top`, `Right.m` overriding: `m()` is `Right.m`."""
    assert [e for e in sample_edges if e.startswith("diamond/Bottom.java::run ->")] == [
        "diamond/Bottom.java::run -> diamond/Right.java::m [calls]"
    ]
