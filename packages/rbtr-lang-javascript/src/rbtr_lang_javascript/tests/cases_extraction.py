"""JavaScript / TypeScript / TSX extraction test cases.

Each `@case` returns test data consumed by `test_extraction.py` via
`pytest-cases`. See the plugin docstring for the source→chunk mapping.
"""

from __future__ import annotations

import pytest
from pytest_cases import case

from rbtr.testing.assertions import Referrer, Target

type SymbolCase = tuple[dict[str, str], list[tuple[str, str, str]]]
type ImportCase = tuple[dict[str, str], dict[str, str | list[dict[str, str]]]]
type MultiImportCase = tuple[dict[str, str], int, list[dict[str, str | list[dict[str, str]]]]]
type MixedCase = tuple[dict[str, str], set[str], list[tuple[str, str]]]

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_js_function_declaration() -> SymbolCase:
    """function greet() {}."""
    return {"input.js": "function greet() {}\n"}, [("function", "greet", "")]


@case(tags=["symbol"])
def case_js_arrow_function() -> SymbolCase:
    """const add = (a, b) => a + b."""
    return {"input.js": "const add = (a, b) => a + b;\n"}, [("function", "add", "")]


@case(tags=["symbol"])
def case_js_arrow_function_block() -> SymbolCase:
    """Arrow function with block body."""
    src = """\
const fetch = () => {
  return data;
};
"""
    return {"input.js": src}, [("function", "fetch", "")]


@case(tags=["symbol"])
def case_js_multiple_functions() -> SymbolCase:
    """Multiple function forms."""
    src = """\
function a() {}
function b() {}
const c = () => {};
"""
    return (
        {"input.js": src},
        [
            ("function", "a", ""),
            ("function", "b", ""),
            ("function", "c", ""),
        ],
    )


@case(tags=["symbol"])
def case_js_class() -> SymbolCase:
    """class User {}."""
    return {"input.js": "class User {}\n"}, [("class", "User", "")]


@case(tags=["symbol"])
def case_js_class_extends() -> SymbolCase:
    """class Admin extends User {}."""
    return {"input.js": "class Admin extends User {}\n"}, [("class", "Admin", "")]


@case(tags=["symbol"])
def case_js_nested_function() -> SymbolCase:
    """A function nested in a function is addressed by the outer function."""
    src = """\
function outer() {
  function inner() {
    return 1;
  }
  return inner;
}
"""
    return {"input.js": src}, [("function", "inner", "outer")]


@case(tags=["import"])
def case_js_import_named_single() -> ImportCase:
    """import { foo } from './models'."""
    return (
        {"input.js": "import { foo } from './models';\n"},
        {"module": "./models", "names": "foo", "bindings": [{"name": "foo", "local": "foo"}]},
    )


@case(tags=["import"])
def case_js_import_named_multiple() -> ImportCase:
    """import { foo, bar } from './models'."""
    return (
        {"input.js": "import { foo, bar } from './models';\n"},
        {
            "module": "./models",
            "names": "foo,bar",
            "bindings": [{"name": "foo", "local": "foo"}, {"name": "bar", "local": "bar"}],
        },
    )


@case(tags=["import"])
def case_js_import_named_parent() -> ImportCase:
    """import from parent directory."""
    return (
        {"input.js": "import { Config } from '../config';\n"},
        {
            "module": "../config",
            "names": "Config",
            "bindings": [{"name": "Config", "local": "Config"}],
        },
    )


@case(tags=["import"])
def case_js_import_named_grandparent() -> ImportCase:
    """import from grandparent directory."""
    return (
        {"input.js": "import { util } from '../../shared/util';\n"},
        {
            "module": "../../shared/util",
            "names": "util",
            "bindings": [{"name": "util", "local": "util"}],
        },
    )


@case(tags=["import"])
def case_js_import_default() -> ImportCase:
    """import React from 'react'."""
    return (
        {"input.js": "import React from 'react';\n"},
        {"module": "react", "names": "React", "bindings": [{"name": "React", "local": "React"}]},
    )


@case(tags=["import"])
def case_js_import_default_relative() -> ImportCase:
    """import App from './App'."""
    return (
        {"input.js": "import App from './App';\n"},
        {"module": "./App", "names": "App", "bindings": [{"name": "App", "local": "App"}]},
    )


@case(tags=["import"])
def case_js_import_namespace() -> ImportCase:
    """import * as utils from '../utils'."""
    return (
        {"input.js": "import * as utils from '../utils';\n"},
        {"module": "../utils", "bindings": [{"name": "../utils", "local": "utils"}]},
    )


@case(tags=["import"])
def case_js_import_side_effect() -> ImportCase:
    """import './styles.css' — side-effect only."""
    return {"input.js": "import './styles.css';\n"}, {"module": "./styles"}


@case(tags=["import"])
def case_js_import_side_effect_no_ext() -> ImportCase:
    """import './polyfills' — no extension."""
    return {"input.js": "import './polyfills';\n"}, {"module": "./polyfills"}


@case(tags=["import"])
def case_js_import_package() -> ImportCase:
    """import express from 'express' — absolute."""
    return (
        {"input.js": "import express from 'express';\n"},
        {
            "module": "express",
            "names": "express",
            "bindings": [{"name": "express", "local": "express"}],
        },
    )


@case(tags=["import"])
def case_js_import_scoped_package() -> ImportCase:
    """import from scoped npm package."""
    return (
        {"input.js": "import { render } from '@testing-library/react';\n"},
        {
            "module": "@testing-library/react",
            "names": "render",
            "bindings": [{"name": "render", "local": "render"}],
        },
    )


@case(tags=["multi_import"])
def case_js_multiple_imports() -> MultiImportCase:
    """Two import statements."""
    src = """\
import React from 'react';
import { useState } from 'react';
"""
    return (
        {"input.js": src},
        2,
        [
            {
                "module": "react",
                "names": "React",
                "bindings": [{"name": "React", "local": "React"}],
            },
            {
                "module": "react",
                "names": "useState",
                "bindings": [{"name": "useState", "local": "useState"}],
            },
        ],
    )


@case(tags=["mixed"])
def case_js_full_module() -> MixedCase:
    """Realistic JS module with JSDoc on every symbol.

    Adding JSDoc here exercises the production-plugin default
    (leading-comment attachment on) against the conventional
    JS documentation style.  The expected-kinds tuple is
    unchanged — this confirms documentation does not disturb
    symbol extraction.  Content assertions for docstring
    presence are covered by `test_docstrings.py`.
    """
    src = """\
import { Model } from './model';

/** Service facade over `Model`. */
class Service {
}

/** Create a new service instance. */
function create() {
}

/** Tear it down. */
const destroy = () => {};
"""
    return {"input.js": src}, {"import", "class", "function"}, []


@case(tags=["symbol"])
def case_ts_function() -> SymbolCase:
    """function greet(): void {}."""
    return {"input.ts": "function greet(): void {}\n"}, [("function", "greet", "")]


@case(tags=["symbol"])
def case_ts_arrow_function() -> SymbolCase:
    """Arrow function with type annotations."""
    return (
        {"input.ts": "const add = (a: number, b: number): number => a + b;\n"},
        [("function", "add", "")],
    )


@case(tags=["symbol"])
def case_ts_class() -> SymbolCase:
    """class Service {}."""
    return {"input.ts": "class Service {}\n"}, [("class", "Service", "")]


@case(tags=["symbol"])
def case_ts_class_generics() -> SymbolCase:
    """class Container<T> {}."""
    return {"input.ts": "class Container<T> {}\n"}, [("class", "Container", "")]


@case(tags=["symbol"])
def case_ts_namespace_function() -> SymbolCase:
    """A function in a TS namespace is addressed by the namespace.

    TS `namespace` is not tracked today (`f` → ""); target "N".
    """
    src = """\
namespace N {
  export function f(): void {}
}
"""
    return {"input.ts": src}, [("function", "f", "N")]


@case(tags=["symbol"])
def case_ts_nested_namespace() -> SymbolCase:
    """Nested TS namespaces compose."""
    src = """\
namespace A {
  export namespace B {
    export function f(): void {}
  }
}
"""
    return {"input.ts": src}, [("function", "f", "A::B")]


@case(tags=["import"])
def case_ts_import_named() -> ImportCase:
    """import { User } from './types'."""
    return (
        {"input.ts": "import { User } from './types';\n"},
        {"module": "./types", "names": "User", "bindings": [{"name": "User", "local": "User"}]},
    )


@case(tags=["import"])
def case_ts_import_type() -> ImportCase:
    """import type { Config } from './config'."""
    return (
        {"input.ts": "import type { Config } from './config';\n"},
        {
            "module": "./config",
            "names": "Config",
            "bindings": [{"name": "Config", "local": "Config"}],
        },
    )


@case(tags=["import"])
def case_ts_import_default() -> ImportCase:
    """import Express from 'express'."""
    return (
        {"input.ts": "import Express from 'express';\n"},
        {
            "module": "express",
            "names": "Express",
            "bindings": [{"name": "Express", "local": "Express"}],
        },
    )


@case(tags=["import"])
def case_ts_import_namespace() -> ImportCase:
    """import * as path from 'path'."""
    return (
        {"input.ts": "import * as path from 'path';\n"},
        {"module": "path", "bindings": [{"name": "path", "local": "path"}]},
    )


@case(tags=["import"])
def case_ts_import_side_effect() -> ImportCase:
    """import './setup'."""
    return {"input.ts": "import './setup';\n"}, {"module": "./setup"}


@case(tags=["mixed"])
def case_ts_full_module() -> MixedCase:
    """Realistic TS module with JSDoc on every symbol.

    Mirrors the JS variant; expected-kinds tuple unchanged.
    Content assertions for docstring presence are in
    `test_docstrings.py`.
    """
    src = """\
import { Model } from './model';

/** Repository over `Model` records. */
class Repository {
}

/** Run a query and return the first row. */
function query(): void {
}
"""
    return {"input.ts": src}, {"import", "class", "function"}, []


@case(tags=["symbol"])
def case_js_module_const() -> SymbolCase:
    """Top-level const."""
    return {"input.js": "const MAX = 100;\n"}, [("variable", "MAX", "")]


@case(tags=["symbol"])
def case_js_exported_const() -> SymbolCase:
    """Exported top-level const."""
    return {"input.js": "export const TIMEOUT = 30;\n"}, [("variable", "TIMEOUT", "")]


@case(tags=["symbol"])
def case_ts_module_const() -> SymbolCase:
    """Top-level annotated const."""
    return {"input.ts": "const MAX: number = 100;\n"}, [("variable", "MAX", "")]


@case(tags=["symbol"])
def case_js_object_destructure() -> SymbolCase:
    """Object destructuring (shorthand)."""
    return {"input.js": "const {a, b} = o;\n"}, [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_js_object_renamed() -> SymbolCase:
    """Object destructuring with rename binds the renamed target."""
    return {"input.js": "const {a: ra} = o;\n"}, [("variable", "ra", "")]


@case(tags=["symbol"])
def case_js_array_destructure() -> SymbolCase:
    """Array destructuring."""
    return {"input.js": "const [x, y] = arr;\n"}, [("variable", "x", ""), ("variable", "y", "")]


@case(tags=["symbol"])
def case_js_object_rest() -> SymbolCase:
    """Object rest element."""
    return (
        {"input.js": "const {a, ...rest} = o;\n"},
        [("variable", "a", ""), ("variable", "rest", "")],
    )


@case(tags=["symbol"])
def case_js_exported_destructure() -> SymbolCase:
    """Exported destructuring."""
    return (
        {"input.js": "export const {a, b} = o;\n"},
        [("variable", "a", ""), ("variable", "b", "")],
    )


@case(tags=["symbol"])
def case_ts_object_destructure() -> SymbolCase:
    """TS object destructuring."""
    return {"input.ts": "const {a, b} = o;\n"}, [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"], marks=_xfail_nested)
def case_js_nested_array_xfail() -> SymbolCase:
    """Nested array destructuring — only the outer level captured today."""
    return (
        {"input.js": "const [a, [b, c]] = x;\n"},
        [("variable", "a", ""), ("variable", "b", ""), ("variable", "c", "")],
    )


@case(tags=["symbol"], marks=_xfail_nested)
def case_js_nested_object_xfail() -> SymbolCase:
    """Nested object destructuring — nothing captured today."""
    return {"input.js": "const {a: {b}} = x;\n"}, [("variable", "b", "")]


type ReferencesCase = tuple[dict[str, str], dict[Target, set[Referrer]]]


@case(tags=["references"])
def case_js_calls_and_bases() -> ReferencesCase:
    """Each call shape, a base by name and by path, and the receivers `this` and `super`.

    `m` is a function of its own, which `x[0].m()`, a call through a value,
    does not reach.
    """
    order = """\
import { b as local } from './ui';
import * as utils from '../utils';
import * as mixins from '../mixins';
import { Base } from '../base';
import { Foo } from '../foo';

class Order extends Base {
  check() {}

  save() {
    super.save();
    this.check();
    utils.log();
    local();
    new Foo();
    a?.b();
    x[0].m();
  }
}

class Plain extends mixins.Logged {}
"""
    utils = """\
export function log() {}

export function m() {}
"""
    base = """\
export class Base {
  save() {}
}
"""
    return (
        {
            "app/order.js": order,
            "app/ui.js": "export function b() {}\n",
            "utils.js": utils,
            "mixins.js": "export class Logged {}\n",
            "base.js": base,
            "foo.js": "export class Foo {}\n",
        },
        {
            ("base.js", "Base"): {("app/order.js", "Order", "inherits")},
            ("mixins.js", "Logged"): {("app/order.js", "Plain", "inherits")},
            ("base.js", "save"): {("app/order.js", "save", "calls")},
            ("app/order.js", "check"): {("app/order.js", "save", "calls")},
            ("utils.js", "log"): {("app/order.js", "save", "calls")},
            ("app/ui.js", "b"): {("app/order.js", "save", "calls")},
            ("foo.js", "Foo"): {("app/order.js", "save", "calls")},
            ("utils.js", "m"): set(),
        },
    )


@case(tags=["references"])
def case_ts_calls_and_bases() -> ReferencesCase:
    """TypeScript's `extends` is a base and `implements` none; type arguments are not the callee."""
    order = """\
import { Base } from './base';
import { Saveable } from './saveable';
import { sum } from './sum';

class Order extends Base implements Saveable {
  items(): number[] {
    return [];
  }

  total(): number {
    return sum<number>(this.items());
  }
}
"""
    total = """\
export function sum<T>(xs: T[]): number {
  return xs.length;
}
"""
    return (
        {
            "order.ts": order,
            "base.ts": "export class Base {}\n",
            "saveable.ts": "export interface Saveable {}\n",
            "sum.ts": total,
        },
        {
            ("base.ts", "Base"): {("order.ts", "Order", "inherits")},
            ("saveable.ts", "Saveable"): set(),
            ("sum.ts", "sum"): {("order.ts", "total", "calls")},
            ("order.ts", "items"): {("order.ts", "total", "calls")},
        },
    )


@case(tags=["references"])
def case_tsx_a_rendered_component_is_a_reference() -> ReferencesCase:
    """`<Toolbar>` and `<Ui.Button />` render components; `<div>` is HTML, not the function `div`."""
    app = """\
import { Toolbar, make, div } from './toolbar';
import * as Ui from './ui';

export function App() {
  return (
    <div>
      <Toolbar items={make()}>
        <Ui.Button />
      </Toolbar>
      <motion.div />
    </div>
  );
}
"""
    toolbar = """\
export function Toolbar(props: object) {
  return null;
}

export function make() {
  return [];
}

export function div() {
  return null;
}
"""
    ui = """\
export function Button() {
  return null;
}
"""
    return (
        {
            "app.tsx": app,
            "toolbar.tsx": toolbar,
            "ui.tsx": ui,
        },
        {
            ("toolbar.tsx", "Toolbar"): {("app.tsx", "App", "calls")},
            ("toolbar.tsx", "make"): {("app.tsx", "App", "calls")},
            ("ui.tsx", "Button"): {("app.tsx", "App", "calls")},
            ("toolbar.tsx", "div"): set(),
        },
    )


@case(tags=["references"])
def case_jsx_a_rendered_component_is_a_reference() -> ReferencesCase:
    """A `.jsx` component's element is a reference as in TSX; `<span />` is HTML."""
    app = """\
import { Panel, span } from './panel';

export function App() {
  return <Panel><span /></Panel>;
}
"""
    panel = """\
export function Panel() {
  return null;
}

export function span() {
  return null;
}
"""
    return (
        {"app.jsx": app, "panel.jsx": panel},
        {
            ("panel.jsx", "Panel"): {("app.jsx", "App", "calls")},
            ("panel.jsx", "span"): set(),
        },
    )


@case(tags=["import"])
def case_js_import_a_dotfile_keeps_its_name() -> ImportCase:
    """import './.eslintrc': a name starting with its only dot has no extension."""
    return {"input.js": "import './.eslintrc';\n"}, {"module": "./.eslintrc"}


@case(tags=["import"])
def case_js_import_drops_only_the_last_extension() -> ImportCase:
    """import './vendor/b.min.js' — the last extension goes, `.min` stays."""
    return {"input.js": "import './vendor/b.min.js';\n"}, {"module": "./vendor/b.min"}


@case(tags=["import"])
def case_js_import_a_dotted_package_reads_its_last_dot_as_an_extension() -> ImportCase:
    """import d from 'lodash.debounce' — the last dot is read as an extension, as for a file."""
    return (
        {"input.js": "import d from 'lodash.debounce';\n"},
        {"module": "lodash", "names": "d", "bindings": [{"name": "d", "local": "d"}]},
    )


@case(tags=["import"])
def case_js_import_a_parent_directory() -> ImportCase:
    """import '../' — a directory, no extension."""
    return {"input.js": "import '../';\n"}, {"module": "../"}


@case(tags=["import"])
def case_js_export_star_reexports_every_name() -> ImportCase:
    """export * from './c' — every name `c` makes available is this file's to offer."""
    return {"input.js": "export * from './c.js';\n"}, {"module": "./c", "names": "*"}


@case(tags=["import"])
def case_js_export_star_as_binds_the_module() -> ImportCase:
    """export * as ns from './n' — binds the module under `ns`."""
    return (
        {"input.js": "export * as ns from './n';\n"},
        {"module": "./n", "bindings": [{"name": "./n", "local": "ns"}]},
    )


@case(tags=["import"])
def case_js_export_names_from_reexports_them() -> ImportCase:
    """export { a, b as bb } from './b' — lists `a` and `b`, offered as `a` and `bb`."""
    return (
        {"input.js": "export { a, b as bb } from './b';\n"},
        {
            "module": "./b",
            "names": "a,b",
            "bindings": [{"name": "a", "local": "a"}, {"name": "b", "local": "bb"}],
        },
    )


@case(tags=["import"])
def case_js_require_binds_the_module() -> ImportCase:
    """const x = require('./x.js') — binds the module under `x`."""
    return (
        {"input.js": "const x = require('./x.js');\n"},
        {"module": "./x", "bindings": [{"name": "./x", "local": "x"}]},
    )


@case(tags=["import"])
def case_js_destructured_require_lists_its_names() -> ImportCase:
    """var { y, z: zz } = require('./y') — lists `y` and `z`, bound as `y` and `zz`."""
    return (
        {"input.js": "var { y, z: zz } = require('./y');\n"},
        {
            "module": "./y",
            "names": "y,z",
            "bindings": [{"name": "y", "local": "y"}, {"name": "z", "local": "zz"}],
        },
    )
