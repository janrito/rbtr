"""What JavaScript's call edges cannot resolve by names alone, each a small project under `samples/calls/`.

- `default-export`: a default import binds the module's default export,
  whatever name the importing file gives it.
- `member-order`: which inherited member a call on `this` reaches. A class
  has one base, so an override hides what it overrides; TypeScript's
  `implements` names an interface, whose signatures no call reaches.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sample_dir(request: pytest.FixtureRequest) -> Path:
    return Path(__file__).parent / "samples" / "calls" / str(request.param)


@pytest.mark.parametrize("sample_dir", ["default-export"], indirect=True)
@pytest.mark.xfail(
    strict=True,
    reason="names only: a default import binds its local name, and the module "
    "defines nothing by that name, so the default export is not reached",
)
def test_a_default_import_reaches_the_default_export(sample_edges: list[str]) -> None:
    """`import Button from "./widget.js"` binds `Widget`, the default export; `Button()` calls it."""
    assert "app.js::run -> widget.js::Widget [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
@pytest.mark.parametrize("ext", ["js", "ts"])
def test_an_override_hides_its_bases_member(sample_edges: list[str], ext: str) -> None:
    """`Special extends Order extends Base`, `Order.save` overriding: `this.save()` is `Order.save`."""
    case = f"override-{ext}"
    assert [e for e in sample_edges if e.startswith(f"{case}/special.{ext}::run ->")] == [
        f"{case}/special.{ext}::run -> {case}/base.{ext}::check [calls]",
        f"{case}/special.{ext}::run -> {case}/order.{ext}::save [calls]",
    ]
    assert f"{case}/order.{ext}::save -> {case}/base.{ext}::save [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["member-order"], indirect=True)
def test_an_implemented_interfaces_signature_is_not_reached(sample_edges: list[str]) -> None:
    """`Order extends Base implements Saver`: `this.save()` is `Base.save`, not `Saver`'s signature."""
    assert [e for e in sample_edges if e.startswith("implements-ts/order.ts::run ->")] == [
        "implements-ts/order.ts::run -> implements-ts/base.ts::save [calls]"
    ]


@pytest.mark.parametrize("sample_dir", ["commonjs"], indirect=True)
def test_a_required_name_reaches_its_definition(sample_edges: list[str]) -> None:
    """`const { log } = require("./util")` then `log()` calls `util.js`'s `log`."""
    assert "app.js::run -> util.js::log [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["commonjs"], indirect=True)
def test_a_reexported_name_reaches_its_definition(sample_edges: list[str]) -> None:
    """`export { log } from "./util.js"` in `index.js`: `log()` imported from it reaches `util.js`'s `log`."""
    assert "start.js::start -> util.js::log [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["commonjs"], indirect=True)
def test_a_required_module_reaches_its_members(sample_edges: list[str]) -> None:
    """`const util = require("./util")` then `util.warn()` calls `util.js`'s `warn`."""
    assert "app.js::run -> util.js::warn [calls]" in sample_edges


@pytest.mark.parametrize("sample_dir", ["commonjs"], indirect=True)
def test_commonjs_edges(sample_edges: list[str]) -> None:
    """Each call reaches only its definition; a required name's variable is no target.

    A whole-module `require` imports every name, as `import * as` does.
    """
    assert sample_edges == [
        'app.js::const util = require("./util"); -> util.js::log [imports]',
        'app.js::const util = require("./util"); -> util.js::warn [imports]',
        'app.js::const { log } = require("./util"); -> util.js::log [imports]',
        "app.js::run -> util.js::log [calls]",
        "app.js::run -> util.js::warn [calls]",
        'index.js::export { log } from "./util.js"; -> util.js::log [imports]',
        "start.js::start -> util.js::log [calls]",
    ]


@pytest.mark.xfail(
    strict=True,
    reason="an import edge reaches definitions in the module it names; index.js "
    "only re-exports log, so the import stops there though the call follows it",
)
@pytest.mark.parametrize("sample_dir", ["commonjs"], indirect=True)
def test_an_import_through_a_reexport_reaches_the_definition(sample_edges: list[str]) -> None:
    """`import { log } from "./index.js"`, which re-exports it, imports `util.js`'s `log`."""
    assert 'start.js::import { log } from "./index.js"; -> util.js::log [imports]' in sample_edges
