"""JavaScript / TypeScript / TSX extraction test cases.

Each `@case` returns test data consumed by `test_extraction.py` via
`pytest-cases`. See the plugin docstring for the source→chunk mapping.
"""

from __future__ import annotations

import pytest
from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_js_function_declaration() -> SymbolCase:
    """function greet() {}."""
    return "javascript", "function greet() {}\n", [("function", "greet", "")]


@case(tags=["symbol"])
def case_js_arrow_function() -> SymbolCase:
    """const add = (a, b) => a + b."""
    return "javascript", "const add = (a, b) => a + b;\n", [("function", "add", "")]


@case(tags=["symbol"])
def case_js_arrow_function_block() -> SymbolCase:
    """Arrow function with block body."""
    src = """\
const fetch = () => {
  return data;
};
"""
    return "javascript", src, [("function", "fetch", "")]


@case(tags=["symbol"])
def case_js_multiple_functions() -> SymbolCase:
    """Multiple function forms."""
    src = """\
function a() {}
function b() {}
const c = () => {};
"""
    return (
        "javascript",
        src,
        [
            ("function", "a", ""),
            ("function", "b", ""),
            ("function", "c", ""),
        ],
    )


@case(tags=["symbol"])
def case_js_class() -> SymbolCase:
    """class User {}."""
    return "javascript", "class User {}\n", [("class", "User", "")]


@case(tags=["symbol"])
def case_js_class_extends() -> SymbolCase:
    """class Admin extends User {}."""
    return "javascript", "class Admin extends User {}\n", [("class", "Admin", "")]


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
    return "javascript", src, [("function", "inner", "outer")]


@case(tags=["import"])
def case_js_import_named_single() -> ImportCase:
    """import { foo } from './models'."""
    return (
        "javascript",
        "import { foo } from './models';\n",
        {"module": "models", "names": "foo", "dots": "1"},
    )


@case(tags=["import"])
def case_js_import_named_multiple() -> ImportCase:
    """import { foo, bar } from './models'."""
    return (
        "javascript",
        "import { foo, bar } from './models';\n",
        {"module": "models", "names": "foo,bar", "dots": "1"},
    )


@case(tags=["import"])
def case_js_import_named_parent() -> ImportCase:
    """import from parent directory."""
    return (
        "javascript",
        "import { Config } from '../config';\n",
        {"module": "config", "names": "Config", "dots": "2"},
    )


@case(tags=["import"])
def case_js_import_named_grandparent() -> ImportCase:
    """import from grandparent directory."""
    return (
        "javascript",
        "import { util } from '../../shared/util';\n",
        {"module": "shared/util", "names": "util", "dots": "3"},
    )


@case(tags=["import"])
def case_js_import_default() -> ImportCase:
    """import React from 'react'."""
    return "javascript", "import React from 'react';\n", {"module": "react", "names": "React"}


@case(tags=["import"])
def case_js_import_default_relative() -> ImportCase:
    """import App from './App'."""
    return (
        "javascript",
        "import App from './App';\n",
        {"module": "App", "names": "App", "dots": "1"},
    )


@case(tags=["import"])
def case_js_import_namespace() -> ImportCase:
    """import * as utils from '../utils'."""
    return (
        "javascript",
        "import * as utils from '../utils';\n",
        {"module": "utils", "names": "utils", "dots": "2"},
    )


@case(tags=["import"])
def case_js_import_side_effect() -> ImportCase:
    """import './styles.css' — side-effect only."""
    return "javascript", "import './styles.css';\n", {"module": "styles", "dots": "1"}


@case(tags=["import"])
def case_js_import_side_effect_no_ext() -> ImportCase:
    """import './polyfills' — no extension."""
    return "javascript", "import './polyfills';\n", {"module": "polyfills", "dots": "1"}


@case(tags=["import"])
def case_js_import_package() -> ImportCase:
    """import express from 'express' — absolute."""
    return (
        "javascript",
        "import express from 'express';\n",
        {"module": "express", "names": "express"},
    )


@case(tags=["import"])
def case_js_import_scoped_package() -> ImportCase:
    """import from scoped npm package."""
    return (
        "javascript",
        "import { render } from '@testing-library/react';\n",
        {"module": "@testing-library/react", "names": "render"},
    )


@case(tags=["multi_import"])
def case_js_multiple_imports() -> MultiImportCase:
    """Two import statements."""
    src = """\
import React from 'react';
import { useState } from 'react';
"""
    return (
        "javascript",
        src,
        2,
        [
            {"module": "react", "names": "React"},
            {"module": "react", "names": "useState"},
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
    return "javascript", src, {"import", "class", "function"}, []


@case(tags=["symbol"])
def case_ts_function() -> SymbolCase:
    """function greet(): void {}."""
    return "typescript", "function greet(): void {}\n", [("function", "greet", "")]


@case(tags=["symbol"])
def case_ts_arrow_function() -> SymbolCase:
    """Arrow function with type annotations."""
    return (
        "typescript",
        "const add = (a: number, b: number): number => a + b;\n",
        [("function", "add", "")],
    )


@case(tags=["symbol"])
def case_ts_class() -> SymbolCase:
    """class Service {}."""
    return "typescript", "class Service {}\n", [("class", "Service", "")]


@case(tags=["symbol"])
def case_ts_class_generics() -> SymbolCase:
    """class Container<T> {}."""
    return "typescript", "class Container<T> {}\n", [("class", "Container", "")]


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
    return "typescript", src, [("function", "f", "N")]


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
    return "typescript", src, [("function", "f", "A::B")]


@case(tags=["import"])
def case_ts_import_named() -> ImportCase:
    """import { User } from './types'."""
    return (
        "typescript",
        "import { User } from './types';\n",
        {"module": "types", "names": "User", "dots": "1"},
    )


@case(tags=["import"])
def case_ts_import_type() -> ImportCase:
    """import type { Config } from './config'."""
    return (
        "typescript",
        "import type { Config } from './config';\n",
        {"module": "config", "names": "Config", "dots": "1"},
    )


@case(tags=["import"])
def case_ts_import_default() -> ImportCase:
    """import Express from 'express'."""
    return (
        "typescript",
        "import Express from 'express';\n",
        {"module": "express", "names": "Express"},
    )


@case(tags=["import"])
def case_ts_import_namespace() -> ImportCase:
    """import * as path from 'path'."""
    return "typescript", "import * as path from 'path';\n", {"module": "path", "names": "path"}


@case(tags=["import"])
def case_ts_import_side_effect() -> ImportCase:
    """import './setup'."""
    return "typescript", "import './setup';\n", {"module": "setup", "dots": "1"}


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
    return "typescript", src, {"import", "class", "function"}, []


@case(tags=["symbol"])
def case_js_module_const() -> SymbolCase:
    """Top-level const."""
    return "javascript", "const MAX = 100;\n", [("variable", "MAX", "")]


@case(tags=["symbol"])
def case_js_exported_const() -> SymbolCase:
    """Exported top-level const."""
    return "javascript", "export const TIMEOUT = 30;\n", [("variable", "TIMEOUT", "")]


@case(tags=["symbol"])
def case_ts_module_const() -> SymbolCase:
    """Top-level annotated const."""
    return "typescript", "const MAX: number = 100;\n", [("variable", "MAX", "")]


@case(tags=["symbol"])
def case_js_object_destructure() -> SymbolCase:
    """Object destructuring (shorthand)."""
    return "javascript", "const {a, b} = o;\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_js_object_renamed() -> SymbolCase:
    """Object destructuring with rename binds the renamed target."""
    return "javascript", "const {a: ra} = o;\n", [("variable", "ra", "")]


@case(tags=["symbol"])
def case_js_array_destructure() -> SymbolCase:
    """Array destructuring."""
    return "javascript", "const [x, y] = arr;\n", [("variable", "x", ""), ("variable", "y", "")]


@case(tags=["symbol"])
def case_js_object_rest() -> SymbolCase:
    """Object rest element."""
    return (
        "javascript",
        "const {a, ...rest} = o;\n",
        [("variable", "a", ""), ("variable", "rest", "")],
    )


@case(tags=["symbol"])
def case_js_exported_destructure() -> SymbolCase:
    """Exported destructuring."""
    return (
        "javascript",
        "export const {a, b} = o;\n",
        [("variable", "a", ""), ("variable", "b", "")],
    )


@case(tags=["symbol"])
def case_ts_object_destructure() -> SymbolCase:
    """TS object destructuring."""
    return "typescript", "const {a, b} = o;\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"], marks=_xfail_nested)
def case_js_nested_array_xfail() -> SymbolCase:
    """Nested array destructuring — only the outer level captured today."""
    return (
        "javascript",
        "const [a, [b, c]] = x;\n",
        [("variable", "a", ""), ("variable", "b", ""), ("variable", "c", "")],
    )


@case(tags=["symbol"], marks=_xfail_nested)
def case_js_nested_object_xfail() -> SymbolCase:
    """Nested object destructuring — nothing captured today."""
    return "javascript", "const {a: {b}} = x;\n", [("variable", "b", "")]
