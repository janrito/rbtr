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


@pytest.fixture(scope="module")
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "calls" / str(request.param)


@pytest.mark.parametrize("sample_dir", ["crate-path"], indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason="a path's head is looked up among the names a file binds, and `crate`, "
    "`super` and modules declared with `mod` are bound by no import",
)
def test_a_call_by_its_full_path_reaches_the_function(sample_edges: list[str]) -> None:
    """`crate::util::log()` calls `log` in `src/util.rs`."""
    assert "src/main.rs::main -> src/util.rs::log [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_inherent_method_precedes_a_traits(sample_edges: list[str]) -> None:
    """`Order` has `save` and implements `Saver` (`save`, `describe`): `self.save()` is `Order::save`, `self.describe()` `Saver::describe`."""
    assert [e for e in sample_edges if e.startswith("src/order.rs::run ->")] == [
        "src/order.rs::run -> src/order.rs::save [calls]",
        "src/order.rs::run -> src/saver.rs::describe [calls]",
    ]


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_a_subtrait_reaches_its_supertraits_method(sample_edges: list[str]) -> None:
    """`trait Titled: Named`: `self.name()` in `Titled::title` is `Named::name`."""
    assert "src/titled.rs::title -> src/named.rs::name [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["mod-declaration"], indirect=True)
def test_a_declared_module_is_bound_by_its_name(sample_edges: list[str]) -> None:
    """`mod util;` binds `util` to `util.rs`: `util::log()` in `main` reaches `log`."""
    assert "src/main.rs::main -> src/util.rs::log [calls]" in sample_edges
