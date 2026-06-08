"""Extraction test cases for all languages.

Each `@case` function returns a tuple of test data consumed by
`test_symbol_extraction.py` and `test_import_extraction.py`
via `@parametrize_with_cases`.

Organisation: one section per language, cases tagged by behavior
(`symbol`, `import`, `multi_import`, `mixed`).
"""

from __future__ import annotations

from pytest_cases import case

from .conftest import skip_unless_grammar

# Return types per tag — each @case function returns one of these.
type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]

# ── Skip markers for optional grammars ───────────────────────────────

_skip_js = skip_unless_grammar("javascript")
_skip_ts = skip_unless_grammar("typescript")
_skip_go = skip_unless_grammar("go")
_skip_rust = skip_unless_grammar("rust")
_skip_java = skip_unless_grammar("java")
_skip_c = skip_unless_grammar("c")
_skip_cpp = skip_unless_grammar("cpp")
_skip_ruby = skip_unless_grammar("ruby")


# ═════════════════════════════════════════════════════════════════════
# Python
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
def case_py_simple_function() -> SymbolCase:
    """Top-level function."""
    src = """\
def hello():
    pass
"""
    return "python", src, [("function", "hello", "")]


@case(tags=["symbol"])
def case_py_function_with_args() -> SymbolCase:
    """Function with parameters."""
    src = """\
def add(a, b):
    return a + b
"""
    return "python", src, [("function", "add", "")]


@case(tags=["symbol"])
def case_py_async_function() -> SymbolCase:
    """Async function."""
    src = """\
async def fetch():
    pass
"""
    return "python", src, [("function", "fetch", "")]


@case(tags=["symbol"])
def case_py_multiple_functions() -> SymbolCase:
    """Multiple functions — all present."""
    src = """\
def foo():
    pass

def bar():
    pass

def baz():
    pass
"""
    return (
        "python",
        src,
        [
            ("function", "foo", ""),
            ("function", "bar", ""),
            ("function", "baz", ""),
        ],
    )


@case(tags=["symbol"])
def case_py_decorated_function() -> SymbolCase:
    """Decorated function."""
    src = """\
@decorator
def wrapped():
    pass
"""
    return "python", src, [("function", "wrapped", "")]


@case(tags=["symbol"])
def case_py_simple_class() -> SymbolCase:
    """Top-level class."""
    src = """\
class Foo:
    pass
"""
    return "python", src, [("class", "Foo", "")]


@case(tags=["symbol"])
def case_py_class_with_bases() -> SymbolCase:
    """Class with inheritance."""
    src = """\
class Bar(Foo, Mixin):
    pass
"""
    return "python", src, [("class", "Bar", "")]


@case(tags=["symbol"])
def case_py_decorated_class() -> SymbolCase:
    """Decorated class."""
    src = """\
@dataclass
class Config:
    name: str
"""
    return "python", src, [("class", "Config", "")]


@case(tags=["symbol"])
def case_py_multiple_classes() -> SymbolCase:
    """Multiple classes."""
    src = """\
class A:
    pass

class B:
    pass
"""
    return "python", src, [("class", "A", ""), ("class", "B", "")]


@case(tags=["symbol"])
def case_py_method_in_class() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Foo:
    def bar(self):
        pass
"""
    return "python", src, [("method", "bar", "Foo")]


@case(tags=["symbol"])
def case_py_multiple_methods() -> SymbolCase:
    """Multiple methods in one class."""
    src = """\
class Svc:
    def start(self):
        pass
    def stop(self):
        pass
"""
    return "python", src, [("method", "start", "Svc"), ("method", "stop", "Svc")]


@case(tags=["symbol"])
def case_py_nested_class_method() -> SymbolCase:
    """Method in nested class scoped to inner class."""
    src = """\
class Outer:
    class Inner:
        def deep(self):
            pass
"""
    return "python", src, [("method", "deep", "Inner")]


@case(tags=["symbol"])
def case_py_top_level_not_method() -> SymbolCase:
    """Function after a class is not scoped."""
    src = """\
class Foo:
    pass

def standalone():
    pass
"""
    return "python", src, [("function", "standalone", "")]


@case(tags=["symbol"])
def case_py_static_method_scoped() -> SymbolCase:
    """Staticmethod still scoped to class."""
    src = """\
class Svc:
    @staticmethod
    def create():
        pass
"""
    return "python", src, [("method", "create", "Svc")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
def case_py_import_bare() -> ImportCase:
    """import os."""
    return "python", "import os\n", {"module": "os"}


@case(tags=["import"])
def case_py_import_dotted() -> ImportCase:
    """import os.path."""
    return "python", "import os.path\n", {"module": "os.path"}


@case(tags=["import"])
def case_py_import_deeply_nested() -> ImportCase:
    """import a.b.c.d."""
    return "python", "import a.b.c.d\n", {"module": "a.b.c.d"}


@case(tags=["import"])
def case_py_from_import_single() -> ImportCase:
    """from pathlib import Path."""
    return "python", "from pathlib import Path\n", {"module": "pathlib", "names": "Path"}


@case(tags=["import"])
def case_py_from_import_multiple() -> ImportCase:
    """from module import multiple names."""
    return (
        "python",
        "from rbtr.index.models import Chunk, Edge\n",
        {"module": "rbtr.index.models", "names": "Chunk,Edge"},
    )


@case(tags=["import"])
def case_py_from_import_aliased() -> ImportCase:
    """Aliased import extracts original name."""
    return (
        "python",
        "from .models import Chunk as C\n",
        {"dots": "1", "module": "models", "names": "Chunk"},
    )


@case(tags=["import"])
def case_py_from_import_multiple_aliased() -> ImportCase:
    """Multiple aliased imports."""
    return (
        "python",
        "from models import Foo as F, Bar as B\n",
        {"module": "models", "names": "Foo,Bar"},
    )


@case(tags=["import"])
def case_py_relative_dot_with_module() -> ImportCase:
    """from .models import Chunk."""
    return (
        "python",
        "from .models import Chunk\n",
        {"dots": "1", "module": "models", "names": "Chunk"},
    )


@case(tags=["import"])
def case_py_relative_dotdot() -> ImportCase:
    """from ..core import engine."""
    return (
        "python",
        "from ..core import engine\n",
        {"dots": "2", "module": "core", "names": "engine"},
    )


@case(tags=["import"])
def case_py_relative_dot_only() -> ImportCase:
    """from . import utils — no module key."""
    return "python", "from . import utils\n", {"dots": "1", "names": "utils"}


@case(tags=["import"])
def case_py_relative_three_dots() -> ImportCase:
    """from ...lib import helper."""
    return (
        "python",
        "from ...lib import helper\n",
        {"dots": "3", "module": "lib", "names": "helper"},
    )


@case(tags=["import"])
def case_py_import_star() -> ImportCase:
    """from os.path import * — no names key."""
    return "python", "from os.path import *\n", {"module": "os.path"}


@case(tags=["import"])
def case_py_import_inside_function() -> ImportCase:
    """Nested import still captured."""
    src = """\
def f():
    import json
"""
    return "python", src, {"module": "json"}


@case(tags=["import"])
def case_py_import_inside_class() -> ImportCase:
    """Import inside class body."""
    src = """\
class C:
    from collections import OrderedDict
"""
    return "python", src, {"module": "collections", "names": "OrderedDict"}


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
def case_py_multiple_imports() -> MultiImportCase:
    """Two bare import statements."""
    src = """\
import os
import sys
"""
    return "python", src, 2, [{"module": "os"}, {"module": "sys"}]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
def case_py_full_module() -> MixedCase:
    """Realistic module with all symbol types and docstrings.

    Docstrings are PEP-257 style on every symbol.  The
    expected-tuple pins symbol extraction invariants; content
    invariants are covered separately by `test_docstrings.py`.
    Adding docs here exercises the realistic shape that
    production Python code has.
    """
    src = '''\
"""Module-level docstring for the Config helper."""

import os
from pathlib import Path


class Config:
    """Runtime configuration."""

    def __init__(self):
        """Initialise with defaults."""
        pass

    def load(self):
        """Load from disk."""
        pass


def main():
    """Entry point."""
    pass
'''
    return (
        "python",
        src,
        {"import", "class", "method", "function"},
        [("__init__", "Config"), ("load", "Config")],
    )


# ═════════════════════════════════════════════════════════════════════
# JavaScript
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_js
def case_js_function_declaration() -> SymbolCase:
    """function greet() {}."""
    return "javascript", "function greet() {}\n", [("function", "greet", "")]


@case(tags=["symbol"])
@_skip_js
def case_js_arrow_function() -> SymbolCase:
    """const add = (a, b) => a + b."""
    return "javascript", "const add = (a, b) => a + b;\n", [("function", "add", "")]


@case(tags=["symbol"])
@_skip_js
def case_js_arrow_function_block() -> SymbolCase:
    """Arrow function with block body."""
    src = """\
const fetch = () => {
  return data;
};
"""
    return "javascript", src, [("function", "fetch", "")]


@case(tags=["symbol"])
@_skip_js
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
@_skip_js
def case_js_class() -> SymbolCase:
    """class User {}."""
    return "javascript", "class User {}\n", [("class", "User", "")]


@case(tags=["symbol"])
@_skip_js
def case_js_class_extends() -> SymbolCase:
    """class Admin extends User {}."""
    return "javascript", "class Admin extends User {}\n", [("class", "Admin", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_js
def case_js_import_named_single() -> ImportCase:
    """import { foo } from './models'."""
    return (
        "javascript",
        "import { foo } from './models';\n",
        {"module": "models", "names": "foo", "dots": "1"},
    )


@case(tags=["import"])
@_skip_js
def case_js_import_named_multiple() -> ImportCase:
    """import { foo, bar } from './models'."""
    return (
        "javascript",
        "import { foo, bar } from './models';\n",
        {"module": "models", "names": "foo,bar", "dots": "1"},
    )


@case(tags=["import"])
@_skip_js
def case_js_import_named_parent() -> ImportCase:
    """import from parent directory."""
    return (
        "javascript",
        "import { Config } from '../config';\n",
        {"module": "config", "names": "Config", "dots": "2"},
    )


@case(tags=["import"])
@_skip_js
def case_js_import_named_grandparent() -> ImportCase:
    """import from grandparent directory."""
    return (
        "javascript",
        "import { util } from '../../shared/util';\n",
        {"module": "shared/util", "names": "util", "dots": "3"},
    )


@case(tags=["import"])
@_skip_js
def case_js_import_default() -> ImportCase:
    """import React from 'react'."""
    return "javascript", "import React from 'react';\n", {"module": "react", "names": "React"}


@case(tags=["import"])
@_skip_js
def case_js_import_default_relative() -> ImportCase:
    """import App from './App'."""
    return (
        "javascript",
        "import App from './App';\n",
        {"module": "App", "names": "App", "dots": "1"},
    )


@case(tags=["import"])
@_skip_js
def case_js_import_namespace() -> ImportCase:
    """import * as utils from '../utils'."""
    return (
        "javascript",
        "import * as utils from '../utils';\n",
        {"module": "utils", "names": "utils", "dots": "2"},
    )


@case(tags=["import"])
@_skip_js
def case_js_import_side_effect() -> ImportCase:
    """import './styles.css' — side-effect only."""
    return "javascript", "import './styles.css';\n", {"module": "styles", "dots": "1"}


@case(tags=["import"])
@_skip_js
def case_js_import_side_effect_no_ext() -> ImportCase:
    """import './polyfills' — no extension."""
    return "javascript", "import './polyfills';\n", {"module": "polyfills", "dots": "1"}


@case(tags=["import"])
@_skip_js
def case_js_import_package() -> ImportCase:
    """import express from 'express' — absolute."""
    return (
        "javascript",
        "import express from 'express';\n",
        {"module": "express", "names": "express"},
    )


@case(tags=["import"])
@_skip_js
def case_js_import_scoped_package() -> ImportCase:
    """import from scoped npm package."""
    return (
        "javascript",
        "import { render } from '@testing-library/react';\n",
        {"module": "@testing-library/react", "names": "render"},
    )


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
@_skip_js
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


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_js
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


# ═════════════════════════════════════════════════════════════════════
# TypeScript
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_ts
def case_ts_function() -> SymbolCase:
    """function greet(): void {}."""
    return "typescript", "function greet(): void {}\n", [("function", "greet", "")]


@case(tags=["symbol"])
@_skip_ts
def case_ts_arrow_function() -> SymbolCase:
    """Arrow function with type annotations."""
    return (
        "typescript",
        "const add = (a: number, b: number): number => a + b;\n",
        [("function", "add", "")],
    )


@case(tags=["symbol"])
@_skip_ts
def case_ts_class() -> SymbolCase:
    """class Service {}."""
    return "typescript", "class Service {}\n", [("class", "Service", "")]


@case(tags=["symbol"])
@_skip_ts
def case_ts_class_generics() -> SymbolCase:
    """class Container<T> {}."""
    return "typescript", "class Container<T> {}\n", [("class", "Container", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_ts
def case_ts_import_named() -> ImportCase:
    """import { User } from './types'."""
    return (
        "typescript",
        "import { User } from './types';\n",
        {"module": "types", "names": "User", "dots": "1"},
    )


@case(tags=["import"])
@_skip_ts
def case_ts_import_type() -> ImportCase:
    """import type { Config } from './config'."""
    return (
        "typescript",
        "import type { Config } from './config';\n",
        {"module": "config", "names": "Config", "dots": "1"},
    )


@case(tags=["import"])
@_skip_ts
def case_ts_import_default() -> ImportCase:
    """import Express from 'express'."""
    return (
        "typescript",
        "import Express from 'express';\n",
        {"module": "express", "names": "Express"},
    )


@case(tags=["import"])
@_skip_ts
def case_ts_import_namespace() -> ImportCase:
    """import * as path from 'path'."""
    return "typescript", "import * as path from 'path';\n", {"module": "path", "names": "path"}


@case(tags=["import"])
@_skip_ts
def case_ts_import_side_effect() -> ImportCase:
    """import './setup'."""
    return "typescript", "import './setup';\n", {"module": "setup", "dots": "1"}


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_ts
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


# ═════════════════════════════════════════════════════════════════════
# Go
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_go
def case_go_function() -> SymbolCase:
    """func hello() {}."""
    src = """\
package main

func hello() {}
"""
    return "go", src, [("function", "hello", "")]


@case(tags=["symbol"])
@_skip_go
def case_go_function_params() -> SymbolCase:
    """func add(a int, b int) int."""
    src = """\
package main

func add(a int, b int) int { return a + b }
"""
    return "go", src, [("function", "add", "")]


@case(tags=["symbol"])
@_skip_go
def case_go_multiple_functions() -> SymbolCase:
    """Multiple functions."""
    src = """\
package main

func foo() {}
func bar() {}
func baz() {}
"""
    return (
        "go",
        src,
        [
            ("function", "foo", ""),
            ("function", "bar", ""),
            ("function", "baz", ""),
        ],
    )


@case(tags=["symbol"])
@_skip_go
def case_go_method() -> SymbolCase:
    """Value receiver method."""
    src = """\
package main

type User struct{}

func (u User) Name() string { return u.name }
"""
    return "go", src, [("method", "Name", "")]


@case(tags=["symbol"])
@_skip_go
def case_go_pointer_method() -> SymbolCase:
    """Pointer receiver method."""
    src = """\
package main

type Svc struct{}

func (s *Svc) Start() {}
"""
    return "go", src, [("method", "Start", "")]


@case(tags=["symbol"])
@_skip_go
def case_go_struct() -> SymbolCase:
    """type User struct."""
    src = """\
package main

type User struct {
    Name string
}
"""
    return "go", src, [("class", "User", "")]


@case(tags=["symbol"])
@_skip_go
def case_go_interface() -> SymbolCase:
    """type Reader interface."""
    src = """\
package main

type Reader interface {
    Read(p []byte) (int, error)
}
"""
    return "go", src, [("class", "Reader", "")]


@case(tags=["symbol"])
@_skip_go
def case_go_type_alias() -> SymbolCase:
    """type ID string."""
    src = """\
package main

type ID string
"""
    return "go", src, [("class", "ID", "")]


@case(tags=["symbol"])
@_skip_go
def case_go_multiple_types() -> SymbolCase:
    """Multiple type declarations."""
    src = """\
package main

type Foo struct{}
type Bar struct{}
"""
    return "go", src, [("class", "Foo", ""), ("class", "Bar", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_go
def case_go_import_single() -> ImportCase:
    """import "fmt"."""
    src = """\
package main
import "fmt"
"""
    return "go", src, {"module": "fmt"}


@case(tags=["import"])
@_skip_go
def case_go_import_nested() -> ImportCase:
    """import "os/exec"."""
    src = """\
package main
import "os/exec"
"""
    return "go", src, {"module": "os/exec"}


@case(tags=["import"])
@_skip_go
def case_go_import_url() -> ImportCase:
    """import "github.com/user/repo"."""
    src = """\
package main
import "github.com/user/repo"
"""
    return "go", src, {"module": "github.com/user/repo"}


@case(tags=["import"])
@_skip_go
def case_go_import_aliased() -> ImportCase:
    """import f "fmt" — alias ignored, module captured."""
    src = """\
package main
import f "fmt"
"""
    return "go", src, {"module": "fmt"}


@case(tags=["multi_import"])
@_skip_go
def case_go_import_grouped() -> MultiImportCase:
    """import ("fmt" "os") — separate chunks per spec."""
    src = """\
package main
import (
    "fmt"
    "os"
)
"""
    return (
        "go",
        src,
        2,
        [{"module": "fmt"}, {"module": "os"}],
    )


@case(tags=["multi_import"])
@_skip_go
def case_go_import_grouped_paths() -> MultiImportCase:
    """Grouped import with nested paths — separate chunks."""
    src = """\
package main
import (
    "fmt"
    "os/exec"
    "net/http"
)
"""
    return (
        "go",
        src,
        3,
        [{"module": "fmt"}, {"module": "os/exec"}, {"module": "net/http"}],
    )


@case(tags=["import"])
@_skip_go
def case_go_import_grouped_single() -> ImportCase:
    """Grouped import with one item."""
    src = """\
package main
import (
    "fmt"
)
"""
    return "go", src, {"module": "fmt"}


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_go
def case_go_full_module() -> MixedCase:
    """Realistic Go module with godoc-style comments.

    Expected-kinds tuple unchanged; content assertions are in
    `test_docstrings.py`.
    """
    src = """\
package main

import (
    "fmt"
    "os"
)

// Config is the runtime configuration for the service.
type Config struct {
    Name string
}

// String returns a human-readable summary of the Config.
func (c Config) String() string { return c.Name }

// main is the entry point for the service.
func main() {
    fmt.Println("hello")
}
"""
    return "go", src, {"import", "class", "method", "function"}, [("String", "")]


# ═════════════════════════════════════════════════════════════════════
# Rust
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_rust
def case_rust_function() -> SymbolCase:
    """fn hello() {}."""
    return "rust", "fn hello() {}\n", [("function", "hello", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_function_params() -> SymbolCase:
    """fn add(a: i32, b: i32) -> i32."""
    return "rust", "fn add(a: i32, b: i32) -> i32 { a + b }\n", [("function", "add", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_pub_function() -> SymbolCase:
    """pub fn visible() {}."""
    return "rust", "pub fn visible() {}\n", [("function", "visible", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_async_function() -> SymbolCase:
    """async fn fetch() {}."""
    return "rust", "async fn fetch() {}\n", [("function", "fetch", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_multiple_functions() -> SymbolCase:
    """Multiple functions."""
    src = """\
fn a() {}
fn b() {}
fn c() {}
"""
    return "rust", src, [("function", "a", ""), ("function", "b", ""), ("function", "c", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_struct() -> SymbolCase:
    """struct User { name: String }."""
    src = """\
struct User {
    name: String,
}
"""
    return "rust", src, [("class", "User", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_unit_struct() -> SymbolCase:
    """struct Marker;"""
    return "rust", "struct Marker;\n", [("class", "Marker", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_tuple_struct() -> SymbolCase:
    """struct Point(f64, f64);"""
    return "rust", "struct Point(f64, f64);\n", [("class", "Point", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_enum() -> SymbolCase:
    """enum Color { Red, Green, Blue }."""
    src = """\
enum Color {
    Red,
    Green,
    Blue,
}
"""
    return "rust", src, [("class", "Color", "")]


@case(tags=["symbol"])
@_skip_rust
def case_rust_method_in_impl() -> SymbolCase:
    """Methods inside impl block scoped to type."""
    src = """\
struct Svc {}
impl Svc {
    fn start(&self) {}
    fn stop(&self) {}
}
"""
    return "rust", src, [("method", "start", "Svc"), ("method", "stop", "Svc")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_rust
def case_rust_import_scoped() -> ImportCase:
    """use std::collections::HashMap."""
    return (
        "rust",
        "use std::collections::HashMap;\n",
        {"module": "std/collections", "names": "HashMap"},
    )


@case(tags=["import"])
@_skip_rust
def case_rust_import_deeply_nested() -> ImportCase:
    """use a::b::c::d::Item."""
    return "rust", "use a::b::c::d::Item;\n", {"module": "a/b/c/d", "names": "Item"}


@case(tags=["import"])
@_skip_rust
def case_rust_import_crate() -> ImportCase:
    """use crate::models::Chunk."""
    return "rust", "use crate::models::Chunk;\n", {"module": "crate/models", "names": "Chunk"}


@case(tags=["import"])
@_skip_rust
def case_rust_import_crate_braces() -> ImportCase:
    """use crate::models::{Chunk, Edge}."""
    return (
        "rust",
        "use crate::models::{Chunk, Edge};\n",
        {"module": "crate/models", "names": "Chunk,Edge"},
    )


@case(tags=["import"])
@_skip_rust
def case_rust_import_super_single() -> ImportCase:
    """use super::utils."""
    return "rust", "use super::utils;\n", {"names": "utils", "dots": "2"}


@case(tags=["import"])
@_skip_rust
def case_rust_import_super_nested() -> ImportCase:
    """use super::helpers::run."""
    return (
        "rust",
        "use super::helpers::run;\n",
        {"module": "helpers", "names": "run", "dots": "2"},
    )


@case(tags=["import"])
@_skip_rust
def case_rust_import_super_super() -> ImportCase:
    """use super::super::common::Config — double super."""
    return (
        "rust",
        "use super::super::common::Config;\n",
        {"dots": "3", "module": "common", "names": "Config"},
    )


@case(tags=["import"])
@_skip_rust
def case_rust_import_use_list() -> ImportCase:
    """use std::io::{Read, Write}."""
    return "rust", "use std::io::{Read, Write};\n", {"module": "std/io", "names": "Read,Write"}


@case(tags=["import"])
@_skip_rust
def case_rust_import_use_list_self() -> ImportCase:
    """use std::io::{self, Read}."""
    return "rust", "use std::io::{self, Read};\n", {"module": "std/io", "names": "self,Read"}


@case(tags=["import"])
@_skip_rust
def case_rust_import_bare() -> ImportCase:
    """use serde;"""
    return "rust", "use serde;\n", {"module": "serde"}


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_rust
def case_rust_full_module() -> MixedCase:
    """Realistic Rust module with `///` doc comments.

    Expected-kinds tuple unchanged; content assertions are in
    `test_docstrings.py`.
    """
    src = """\
use std::collections::HashMap;
use crate::models::Config;

/// Application state.
struct App {
    config: Config,
}

/// Lifecycle status.
enum Status {
    Running,
    Stopped,
}

/// Methods for the `App` type.
impl App {
    /// Construct a new instance from a loaded `Config`.
    fn new(config: Config) -> Self {
        App { config }
    }
    /// Run the main loop.
    fn run(&self) {}
}

/// Entry point.
fn main() {}
"""
    return "rust", src, {"import", "class", "method", "function"}, [("new", "App"), ("run", "App")]


# ═════════════════════════════════════════════════════════════════════
# Java
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_java
def case_java_class() -> SymbolCase:
    """class User {}."""
    return "java", "class User {}\n", [("class", "User", "")]


@case(tags=["symbol"])
@_skip_java
def case_java_public_class() -> SymbolCase:
    """public class App {}."""
    return "java", "public class App {}\n", [("class", "App", "")]


@case(tags=["symbol"])
@_skip_java
def case_java_class_extends() -> SymbolCase:
    """class Admin extends User {}."""
    return "java", "class Admin extends User {}\n", [("class", "Admin", "")]


@case(tags=["symbol"])
@_skip_java
def case_java_class_implements() -> SymbolCase:
    """class UserService implements Service {}."""
    return "java", "class UserService implements Service {}\n", [("class", "UserService", "")]


@case(tags=["symbol"])
@_skip_java
def case_java_multiple_classes() -> SymbolCase:
    """Multiple classes."""
    src = """\
class Foo {}
class Bar {}
"""
    return "java", src, [("class", "Foo", ""), ("class", "Bar", "")]


@case(tags=["symbol"])
@_skip_java
def case_java_method_in_class() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Service {
    void process() {}
}
"""
    return "java", src, [("method", "process", "Service")]


@case(tags=["symbol"])
@_skip_java
def case_java_multiple_methods() -> SymbolCase:
    """Multiple methods."""
    src = """\
class Svc {
    void start() {}
    void stop() {}
}
"""
    return "java", src, [("method", "start", "Svc"), ("method", "stop", "Svc")]


@case(tags=["symbol"])
@_skip_java
def case_java_static_method() -> SymbolCase:
    """Static method still scoped."""
    src = """\
class Factory {
    static Object create() { return null; }
}
"""
    return "java", src, [("method", "create", "Factory")]


@case(tags=["symbol"])
@_skip_java
def case_java_method_params() -> SymbolCase:
    """Method with parameters."""
    src = """\
class Calc {
    int add(int a, int b) { return a + b; }
}
"""
    return "java", src, [("method", "add", "Calc")]


@case(tags=["symbol"])
@_skip_java
def case_java_nested_class() -> SymbolCase:
    """Nested class scoped to outer, method scoped to inner."""
    src = """\
class Outer {
    class Inner {
        void deep() {}
    }
}
"""
    return (
        "java",
        src,
        [("class", "Outer", ""), ("class", "Inner", "Outer"), ("method", "deep", "Inner")],
    )


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_java
def case_java_import_class() -> ImportCase:
    """import java.util.HashMap."""
    return "java", "import java.util.HashMap;\n", {"module": "java.util.HashMap"}


@case(tags=["import"])
@_skip_java
def case_java_import_deeply_nested() -> ImportCase:
    """import com.example.app.models.User."""
    return (
        "java",
        "import com.example.app.models.User;\n",
        {"module": "com.example.app.models.User"},
    )


@case(tags=["import"])
@_skip_java
def case_java_import_static() -> ImportCase:
    """import static org.junit.Assert.assertEquals."""
    return (
        "java",
        "import static org.junit.Assert.assertEquals;\n",
        {"module": "org.junit.Assert.assertEquals"},
    )


@case(tags=["import"])
@_skip_java
def case_java_import_static_method() -> ImportCase:
    """import static java.util.Collections.sort."""
    return (
        "java",
        "import static java.util.Collections.sort;\n",
        {"module": "java.util.Collections.sort"},
    )


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
@_skip_java
def case_java_multiple_imports() -> MultiImportCase:
    """Two import statements."""
    src = """\
import java.util.List;
import java.util.Map;
"""
    return (
        "java",
        src,
        2,
        [
            {"module": "java.util.List"},
            {"module": "java.util.Map"},
        ],
    )


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_java
def case_java_full_class() -> MixedCase:
    """Realistic Java class with Javadoc on every member.

    Expected tuple unchanged; content assertions in
    `test_docstrings.py`.
    """
    src = """\
import java.util.List;
import java.util.ArrayList;

/** Tracks registered user names. */
public class UserService {
    private List<String> names;

    /** Append a new name to the registry. */
    public void addName(String name) {
        names.add(name);
    }

    /** Return the current list of names. */
    public List<String> getNames() {
        return names;
    }
}
"""
    return (
        "java",
        src,
        {"import", "class", "method"},
        [("addName", "UserService"), ("getNames", "UserService")],
    )


# ═════════════════════════════════════════════════════════════════════
# Bash
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
def case_bash_function_keyword() -> SymbolCase:
    """function deploy { ... }."""
    src = """\
function deploy {
    echo deploying
}
"""
    return "bash", src, [("function", "deploy", "")]


@case(tags=["symbol"])
def case_bash_function_keyword_parens() -> SymbolCase:
    """function deploy() { ... }."""
    src = """\
function deploy() {
    echo deploying
}
"""
    return "bash", src, [("function", "deploy", "")]


@case(tags=["symbol"])
def case_bash_function_posix() -> SymbolCase:
    """deploy() { ... } — POSIX syntax."""
    src = """\
deploy() {
    echo deploying
}
"""
    return "bash", src, [("function", "deploy", "")]


@case(tags=["symbol"])
def case_bash_multiple_functions() -> SymbolCase:
    """Multiple shell functions."""
    src = """\
function setup {
    echo setup
}

function teardown {
    echo teardown
}

run() {
    echo run
}
"""
    return (
        "bash",
        src,
        [
            ("function", "setup", ""),
            ("function", "teardown", ""),
            ("function", "run", ""),
        ],
    )


@case(tags=["symbol"])
def case_bash_function_local_vars() -> SymbolCase:
    """Function with local variables."""
    src = """\
setup() {
    local dir="/tmp"
    mkdir -p "$dir"
}
"""
    return "bash", src, [("function", "setup", "")]


@case(tags=["symbol"])
def case_bash_function_conditionals() -> SymbolCase:
    """Function with conditionals."""
    src = """\
check() {
    if [ -f /tmp/x ]; then
        echo yes
    fi
}
"""
    return "bash", src, [("function", "check", "")]


# ── Mixed ───────────────────────────────────────────────────────────


@case(tags=["mixed"])
def case_bash_full_script() -> MixedCase:
    """Realistic shell script with doc comments on every
    function.  Expected-kinds tuple pins symbol extraction;
    content invariants are in `test_docstrings.py`.
    """
    src = """\
#!/bin/bash

# Deploy the current build to the given environment.
deploy() {
    local env="$1"
    echo "deploying to $env"
}

# Roll back the last deploy.
rollback() {
    echo "rolling back"
}
"""
    return "bash", src, {"function"}, []


# ═════════════════════════════════════════════════════════════════════
# C
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_c
def case_c_function_basic() -> SymbolCase:
    """int add(int a, int b)."""
    return "c", "int add(int a, int b) { return a + b; }\n", [("function", "add", "")]


@case(tags=["symbol"])
@_skip_c
def case_c_function_void() -> SymbolCase:
    """void do_stuff(void)."""
    return "c", "void do_stuff(void) { }\n", [("function", "do_stuff", "")]


@case(tags=["symbol"])
@_skip_c
def case_c_function_static() -> SymbolCase:
    """static int helper(void)."""
    return "c", "static int helper(void) { return 1; }\n", [("function", "helper", "")]


@case(tags=["symbol"])
@_skip_c
def case_c_multiple_functions() -> SymbolCase:
    """Multiple C functions."""
    src = """\
int foo(void) { return 0; }
void bar(void) { }
"""
    return "c", src, [("function", "foo", ""), ("function", "bar", "")]


@case(tags=["symbol"])
@_skip_c
def case_c_struct() -> SymbolCase:
    """struct Node."""
    return "c", "struct Node { int value; };\n", [("class", "Node", "")]


@case(tags=["symbol"])
@_skip_c
def case_c_enum() -> SymbolCase:
    """enum Color."""
    return "c", "enum Color { RED, GREEN, BLUE };\n", [("class", "Color", "")]


@case(tags=["symbol"])
@_skip_c
def case_c_typedef_struct() -> SymbolCase:
    """typedef struct { ... } Point."""
    return "c", "typedef struct { int x; int y; } Point;\n", [("class", "Point", "")]


@case(tags=["symbol"])
@_skip_c
def case_c_no_scope() -> SymbolCase:
    """C functions are never scoped — no classes."""
    src = """\
struct S { int x; };
int func(void) { return 0; }
"""
    return "c", src, [("function", "func", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_c
def case_c_include_system() -> ImportCase:
    """#include <stdio.h>."""
    return "c", "#include <stdio.h>\n", {"module": "stdio.h"}


@case(tags=["import"])
@_skip_c
def case_c_include_local() -> ImportCase:
    """#include "mylib.h"."""
    return "c", '#include "mylib.h"\n', {"module": "mylib.h"}


@case(tags=["import"])
@_skip_c
def case_c_include_nested_path() -> ImportCase:
    """#include "utils/helpers.h"."""
    return "c", '#include "utils/helpers.h"\n', {"module": "utils/helpers.h"}


@case(tags=["import"])
@_skip_c
def case_c_include_system_nested() -> ImportCase:
    """#include <sys/types.h>."""
    return "c", "#include <sys/types.h>\n", {"module": "sys/types.h"}


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
@_skip_c
def case_c_multiple_includes() -> MultiImportCase:
    """Two include directives."""
    src = """\
#include <stdlib.h>
#include "local.h"
"""
    return "c", src, 2, [{"module": "stdlib.h"}, {"module": "local.h"}]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_c
def case_c_full_file() -> MixedCase:
    """Realistic C file with Doxygen comments on every symbol.

    Expected-kinds tuple unchanged.
    """
    src = """\
#include <stdio.h>
#include "utils.h"

/** Runtime configuration for the parser. */
struct Config {
    int timeout;
    int retries;
};

/** Return status for parse operations. */
enum Status { OK, ERR };

/** Parse a config file from disk. */
int parse_config(const char *path) {
    return 0;
}

/** Release parser-owned resources. */
static void cleanup(void) {
}
"""
    return "c", src, {"import", "class", "function"}, []


# ═════════════════════════════════════════════════════════════════════
# C++
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_free_function_void() -> SymbolCase:
    """void greet()."""
    return "cpp", "void greet() { }\n", [("function", "greet", "")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_free_function_returning() -> SymbolCase:
    """int compute(int x)."""
    return "cpp", "int compute(int x) { return x; }\n", [("function", "compute", "")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_multiple_functions() -> SymbolCase:
    """Multiple C++ functions."""
    src = """\
int foo() { return 0; }
void bar() { }
"""
    return "cpp", src, [("function", "foo", ""), ("function", "bar", "")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_class() -> SymbolCase:
    """class Shape."""
    return "cpp", "class Shape { };\n", [("class", "Shape", "")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_struct() -> SymbolCase:
    """struct Point."""
    return "cpp", "struct Point { double x; double y; };\n", [("class", "Point", "")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_enum_class() -> SymbolCase:
    """enum class Color."""
    return "cpp", "enum class Color { Red, Green, Blue };\n", [("class", "Color", "")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_class_inheritance() -> SymbolCase:
    """class Derived : public Base."""
    src = """\
class Base { };
class Derived : public Base { };
"""
    return "cpp", src, [("class", "Base", ""), ("class", "Derived", "")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_class_method() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Foo {
public:
    void bar() { }
};
"""
    return "cpp", src, [("method", "bar", "Foo")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_struct_method() -> SymbolCase:
    """Method scoped to struct."""
    return "cpp", "struct Vec { void push(int v) { } };\n", [("method", "push", "Vec")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_multiple_methods() -> SymbolCase:
    """Multiple methods in one class."""
    src = """\
class Calculator {
public:
    int add(int a, int b) { return a + b; }
    int sub(int a, int b) { return a - b; }
};
"""
    return "cpp", src, [("method", "add", "Calculator"), ("method", "sub", "Calculator")]


@case(tags=["symbol"])
@_skip_cpp
def case_cpp_free_function_not_scoped() -> SymbolCase:
    """Function after class is not scoped."""
    src = """\
class C { };
void standalone() { }
"""
    return "cpp", src, [("function", "standalone", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_cpp
def case_cpp_include_system() -> ImportCase:
    """#include <iostream>."""
    return "cpp", "#include <iostream>\n", {"module": "iostream"}


@case(tags=["import"])
@_skip_cpp
def case_cpp_include_local() -> ImportCase:
    """#include "myheader.h"."""
    return "cpp", '#include "myheader.h"\n', {"module": "myheader.h"}


@case(tags=["import"])
@_skip_cpp
def case_cpp_include_nested() -> ImportCase:
    """#include <boost/optional.hpp>."""
    return "cpp", "#include <boost/optional.hpp>\n", {"module": "boost/optional.hpp"}


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_cpp
def case_cpp_full_file() -> MixedCase:
    """Realistic C++ file with Doxygen comments.

    Expected tuple unchanged.
    """
    src = """\
#include <vector>
#include "config.h"

/** Execution engine for the pipeline. */
class Engine {
public:
    void start() { }
    void stop() { }
};

/** Options passed to `Engine::start`. */
struct Options {
    int timeout;
};

/** Execution mode — `Fast` skips integrity checks. */
enum class Mode { Fast, Safe };

/** Drive the engine through a single cycle. */
void run(Engine& e) {
    e.start();
}
"""
    return (
        "cpp",
        src,
        {"import", "class", "method", "function"},
        [("start", "Engine"), ("stop", "Engine")],
    )


# ═════════════════════════════════════════════════════════════════════
# Ruby
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_method_no_args() -> SymbolCase:
    """def greet — top-level function."""
    src = """\
def greet
  puts "hello"
end
"""
    return "ruby", src, [("function", "greet", "")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_method_with_args() -> SymbolCase:
    """def add(a, b) — top-level function."""
    src = """\
def add(a, b)
  a + b
end
"""
    return "ruby", src, [("function", "add", "")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_multiple_functions() -> SymbolCase:
    """Multiple top-level functions."""
    src = """\
def foo
  1
end

def bar
  2
end
"""
    return "ruby", src, [("function", "foo", ""), ("function", "bar", "")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_class() -> SymbolCase:
    """class Shape."""
    src = """\
class Shape
end
"""
    return "ruby", src, [("class", "Shape", "")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_module() -> SymbolCase:
    """module Utils."""
    src = """\
module Utils
end
"""
    return "ruby", src, [("class", "Utils", "")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_class_superclass() -> SymbolCase:
    """class Circle < Shape."""
    src = """\
class Circle < Shape
end
"""
    return "ruby", src, [("class", "Circle", "")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_method_scoped_to_class() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Foo
  def bar
    1
  end
end
"""
    return "ruby", src, [("method", "bar", "Foo")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_singleton_method() -> SymbolCase:
    """def self.build — singleton method."""
    src = """\
class Factory
  def self.build
    new
  end
end
"""
    return "ruby", src, [("method", "build", "Factory")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_method_scoped_to_module() -> SymbolCase:
    """Method scoped to module."""
    src = """\
module Helpers
  def format(s)
    s.strip
  end
end
"""
    return "ruby", src, [("method", "format", "Helpers")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_top_level_not_scoped() -> SymbolCase:
    """Function after class is not scoped."""
    src = """\
class C
end

def standalone
  1
end
"""
    return "ruby", src, [("function", "standalone", "")]


@case(tags=["symbol"])
@_skip_ruby
def case_ruby_nested_class_in_module() -> SymbolCase:
    """Class nested in module, method scoped to inner class."""
    src = """\
module Utils
  class Parser
    def parse(input)
      input
    end
  end
end
"""
    return (
        "ruby",
        src,
        [("class", "Utils", ""), ("class", "Parser", "Utils"), ("method", "parse", "Parser")],
    )


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
@_skip_ruby
def case_ruby_require_simple() -> ImportCase:
    """require "json"."""
    return "ruby", 'require "json"\n', {"module": "json"}


@case(tags=["import"])
@_skip_ruby
def case_ruby_require_nested() -> ImportCase:
    """require "net/http"."""
    return "ruby", 'require "net/http"\n', {"module": "net/http"}


@case(tags=["import"])
@_skip_ruby
def case_ruby_require_relative() -> ImportCase:
    """require_relative "helpers"."""
    return "ruby", 'require_relative "helpers"\n', {"module": "helpers", "dots": "1"}


@case(tags=["import"])
@_skip_ruby
def case_ruby_require_relative_nested() -> ImportCase:
    """require_relative "lib/utils"."""
    return "ruby", 'require_relative "lib/utils"\n', {"module": "lib/utils", "dots": "1"}


@case(tags=["import"])
@_skip_ruby
def case_ruby_require_empty_string() -> ImportCase:
    """require "" — empty string returns empty metadata.

    Covers `ruby.py` lines 53 and 70.
    """
    return "ruby", 'require ""\n', {}


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
@_skip_ruby
def case_ruby_multiple_requires() -> MultiImportCase:
    """require + require_relative."""
    src = """\
require "json"
require_relative "helpers"
"""
    return "ruby", src, 2, [{"module": "json"}, {"module": "helpers", "dots": "1"}]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
@_skip_ruby
def case_ruby_full_file() -> MixedCase:
    """Realistic Ruby file with doc comments on top-level
    declarations.  Comments inside the class body are not
    attached to their methods by the current Ruby grammar (see
    note in `case_docstrings.py`), so only the top-level
    module, class, and `main` carry docs here.  Expected tuple
    unchanged.
    """
    src = """\
require "json"
require_relative "config"

# Application namespace for the service.
module App
  # Server runs the request loop.
  class Server
    def start
      puts "starting"
    end

    def stop
      puts "stopping"
    end

    def self.default
      new
    end
  end
end

# Entry point used by bin/app.
def main
  server = App::Server.new
  server.start
end
"""
    return (
        "ruby",
        src,
        {"import", "class", "method", "function"},
        [("start", "Server"), ("stop", "Server"), ("default", "Server")],
    )


# ══════════════��══════════════════════════════════════════════════════
# Markdown
# ══════════���═════════════════════════════════════════════════════���════


@case(tags=["symbol"])
def case_md_splits_by_heading() -> SymbolCase:
    """Markdown heading hierarchy."""
    src = """\
# Title

Intro text.

## Section A

Body A.

## Section B

Body B.
"""
    return (
        "markdown",
        src,
        [
            ("doc_section", "Title", ""),
            ("doc_section", "Section A", "Title"),
            ("doc_section", "Section B", "Title"),
        ],
    )


@case(tags=["symbol"])
def case_md_scope_chain() -> SymbolCase:
    """Nested heading scope."""
    src = """\
# Top

## Mid

### Deep

Content here.
"""
    return "markdown", src, [("doc_section", "Deep", "Top > Mid")]


@case(tags=["symbol"])
def case_md_empty() -> SymbolCase:
    """Empty markdown produces no chunks."""
    return "markdown", "", []


# ══════════════════════════════��══════════════════════════════════════
# RST
# ═════���════════════��═══════════════════════════════════��══════════════


@case(tags=["symbol"])
def case_rst_splits_by_heading() -> SymbolCase:
    """RST heading hierarchy."""
    src = """\
Title
=====

Intro text.

Section A
---------

Body A.

Section B
---------

Body B.
"""
    return (
        "rst",
        src,
        [
            ("doc_section", "Title", "Title"),
            ("doc_section", "Section B", "Title"),
        ],
    )


@case(tags=["symbol"])
def case_rst_scope_chain() -> SymbolCase:
    """RST nested scope from adornment order."""
    src = """\
Top
===

Mid
---

Deep
^^^

Content here.
"""
    return "rst", src, [("doc_section", "Deep", "Top > Mid")]
    # Note: RST scope includes self for non-final sections due to
    # scope_stack push order. Deep is the final section so it
    # happens to show only parent scope.


@case(tags=["symbol"])
def case_rst_empty() -> SymbolCase:
    """Empty RST produces no chunks."""
    return "rst", "", []


# ═════════════════════��════════════════════════════════════��══════════
# JSON
# ════════════════════════════════════════════════���════════════════════


@case(tags=["symbol"])
def case_json_top_level_keys() -> SymbolCase:
    """JSON splits by top-level keys."""
    src = """\
{
  "name": "my-project",
  "version": "1.0.0",
  "dependencies": {
    "foo": "^1.0"
  }
}
"""
    return (
        "json",
        src,
        [
            ("doc_section", "name", ""),
            ("doc_section", "version", ""),
            ("doc_section", "dependencies", ""),
        ],
    )


@case(tags=["symbol"])
def case_json_empty() -> SymbolCase:
    """Empty JSON produces no chunks."""
    return "json", "", []


# ══════════════════════���══════════════════════════════════════════════
# CSS
# ════════════════���══════════════════════════��═════════════════════════


@case(tags=["symbol"])
def case_css_rule_sets() -> SymbolCase:
    """CSS splits by rule sets."""
    src = """\
body {
  color: #333;
}

.header {
  background: blue;
}
"""
    return (
        "css",
        src,
        [
            ("doc_section", "body", ""),
            ("doc_section", ".header", ""),
        ],
    )


@case(tags=["symbol"])
def case_css_empty() -> SymbolCase:
    """Empty CSS produces no chunks."""
    return "css", "", []


# ══════════════════════════════════════════════════��══════════════════
# TOML
# ════════════��════════════════════════════════════════════════════════


@case(tags=["symbol"])
def case_toml_splits_by_table() -> SymbolCase:
    """TOML splits by tables with dotted scope."""
    src = """\
[project]
name = "rbtr"

[tool.ruff]
line-length = 99
"""
    return (
        "toml",
        src,
        [
            ("doc_section", "project", ""),
            ("doc_section", "ruff", "tool"),
        ],
    )


@case(tags=["symbol"])
def case_toml_empty() -> SymbolCase:
    """Empty TOML produces no chunks."""
    return "toml", "", []


# ═��═════════════════════════════════════════��═════════════════════════
# YAML
# ═══════════════════════════════════════════════════════════��═════════


@case(tags=["symbol"])
def case_yaml_top_level_keys() -> SymbolCase:
    """YAML splits by top-level mapping keys."""
    src = """\
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
"""
    return (
        "yaml",
        src,
        [
            ("doc_section", "name", ""),
            ("doc_section", "on", ""),
            ("doc_section", "jobs", ""),
        ],
    )


@case(tags=["symbol"])
def case_yaml_empty() -> SymbolCase:
    """Empty YAML produces no chunks."""
    return "yaml", "", []


# ═════���══════���════════════════════════════════════════════════════════
# HCL
# ═══��═══════════════════════════════════════���═════════════════════════


@case(tags=["symbol"])
def case_hcl_splits_by_blocks() -> SymbolCase:
    """HCL splits by top-level blocks."""
    src = """\
resource "aws_instance" "web" {
  ami = "ami-12345"
}

variable "region" {
  default = "us-east-1"
}
"""
    return (
        "hcl",
        src,
        [
            ("doc_section", "resource aws_instance web", ""),
            ("doc_section", "variable region", ""),
        ],
    )


@case(tags=["symbol"])
def case_hcl_empty() -> SymbolCase:
    """Empty HCL produces no chunks."""
    return "hcl", "", []


# ═════════════���═══════════════════════════════════════════════════════
# HTML
# ════════════════════════════════════════════════════════════════════��


@case(tags=["symbol"])
def case_html_body_elements() -> SymbolCase:
    """HTML splits by body elements."""
    src = """\
<html>
<body>
  <h1>Title</h1>
  <p>Content</p>
</body>
</html>
"""
    return (
        "html",
        src,
        [
            ("doc_section", "h1", ""),
            ("doc_section", "p", ""),
        ],
    )


@case(tags=["symbol"])
def case_html_empty() -> SymbolCase:
    """Empty HTML produces no chunks."""
    return "html", "", []
