"""Extraction test cases for all languages.

Each `@case` function returns a tuple of test data consumed by
`test_symbol_extraction.py` and `test_import_extraction.py`
via `@parametrize_with_cases`.

Organisation: one section per language, cases tagged by behavior
(`symbol`, `import`, `multi_import`, `mixed`).
"""

from __future__ import annotations

import pytest
from pytest_cases import case

# Return types per tag — each @case function returns one of these.
type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]


# ═════════════════════════════════════════════════════════════════════
# Python
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


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
    """Method in a nested class carries the full class path."""
    src = """\
class Outer:
    class Inner:
        def deep(self):
            pass
"""
    return "python", src, [("method", "deep", "Outer::Inner")]


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


@case(tags=["symbol"])
def case_py_closure_in_function() -> SymbolCase:
    """A closure is addressed by its enclosing function and stays a function.

    Function nesting must contribute to the address (inner `handler`
    → scope "make_adder"), and a function nested in a function must
    remain kind=function — it is not a method.
    """
    src = """\
def make_adder(n):
    def handler():
        return n + 1
    return handler
"""
    return "python", src, [("function", "handler", "make_adder")]


@case(tags=["symbol"])
def case_py_closure_in_method_stays_function() -> SymbolCase:
    """A function nested in a method is addressed fully and NOT promoted.

    The enclosing scope path is "Svc::start" and `cb` must stay
    kind=function — promotion follows the nearest *enclosing scope
    node's* type (a class), not merely a non-empty scope.
    """
    src = """\
class Svc:
    def start(self):
        def cb():
            return 1
        return cb
"""
    return "python", src, [("function", "cb", "Svc::start")]


@case(tags=["symbol"])
def case_py_repeated_nested_name() -> SymbolCase:
    """Repeated scope names compose without collapsing.

    Three classes all named `Node` nest; the innermost method's path
    keeps every level (`Node::Node::Node`), not a deduplicated one.
    """
    src = """\
class Node:
    class Node:
        class Node:
            def visit(self):
                pass
"""
    return "python", src, [("method", "visit", "Node::Node::Node")]


@case(tags=["symbol"])
def case_py_shadowed_closure() -> SymbolCase:
    """A closure shadowing its enclosing function's name stays distinct.

    The inner `process` is addressed `process` (its outer namesake);
    the two share a name but differ in scope, so addressing keeps
    them apart. The inner one is a function, not a method.
    """
    src = """\
def process():
    def process():
        return 1
    return process
"""
    return "python", src, [("function", "process", "process"), ("function", "process", "")]


@case(tags=["symbol"])
def case_py_mixed_deep_nesting() -> SymbolCase:
    """Mixed class/method/closure/class chain composes the full path.

    `deep` sits in a local class, in a closure, in a method, in a
    class — so its path mixes every scope kind. Promotion follows the
    *nearest* scope: `helper` (in a method) is a function, `deep` (in
    a class) is a method.
    """
    src = """\
class Outer:
    def run(self):
        def helper():
            class Local:
                def deep(self):
                    pass
            return Local
        return helper
"""
    return (
        "python",
        src,
        [
            ("method", "run", "Outer"),
            ("function", "helper", "Outer::run"),
            ("class", "Local", "Outer::run::helper"),
            ("method", "deep", "Outer::run::helper::Local"),
        ],
    )


@case(tags=["symbol"])
def case_py_method_named_like_class() -> SymbolCase:
    """A method whose name equals its class is addressed `Node::`, not merged.

    The method `Node` inside class `Node` is `(method, Node, Node)` —
    name and scope segment coincide but are different objects; the
    address keeps both.
    """
    src = """\
class Node:
    def Node(self):
        return 1
"""
    return "python", src, [("class", "Node", ""), ("method", "Node", "Node")]


@case(tags=["symbol"])
def case_py_method_named_like_nested_class() -> SymbolCase:
    """Name equal to a *repeated* scope segment still composes fully.

    A class `Node` in a class `Node` with a method `Node`: the method
    is `(method, Node, Node::Node)` and the inner class is
    `(class, Node, Node)`.
    """
    src = """\
class Node:
    class Node:
        def Node(self):
            return 1
"""
    return "python", src, [("class", "Node", "Node"), ("method", "Node", "Node::Node")]


@case(tags=["symbol"])
def case_py_function_and_class_same_name() -> SymbolCase:
    """A function and a class sharing a name at module scope both extract.

    Different objects (`function` and `class`) collide on identity
    `(name, scope)` = `(Cache, "")` — the residual same-scope
    collision addressing cannot split (kind is not part of identity).
    Both must appear; the diff's content-set backstop keeps them apart.
    """
    src = """\
def Cache():
    return None

class Cache:
    pass
"""
    return "python", src, [("function", "Cache", ""), ("class", "Cache", "")]


@case(tags=["symbol"])
def case_py_module_constant() -> SymbolCase:
    """Module-level constant."""
    src = """\
MAX_SIZE = 100
"""
    return "python", src, [("variable", "MAX_SIZE", "")]


@case(tags=["symbol"])
def case_py_module_singleton() -> SymbolCase:
    """Module-level lowercase singleton (the motivating case)."""
    src = """\
config = Config()
"""
    return "python", src, [("variable", "config", "")]


@case(tags=["symbol"])
def case_py_annotated_module_var() -> SymbolCase:
    """Module-level annotated assignment."""
    src = """\
TIMEOUT: int = 30
"""
    return "python", src, [("variable", "TIMEOUT", "")]


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
def case_go_function() -> SymbolCase:
    """func hello() {}."""
    src = """\
package main

func hello() {}
"""
    return "go", src, [("function", "hello", "")]


@case(tags=["symbol"])
def case_go_function_params() -> SymbolCase:
    """func add(a int, b int) int."""
    src = """\
package main

func add(a int, b int) int { return a + b }
"""
    return "go", src, [("function", "add", "")]


@case(tags=["symbol"])
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
def case_go_method() -> SymbolCase:
    """Value receiver method."""
    src = """\
package main

type User struct{}

func (u User) Name() string { return u.name }
"""
    return "go", src, [("method", "Name", "User")]


@case(tags=["symbol"])
def case_go_pointer_method() -> SymbolCase:
    """Pointer receiver method."""
    src = """\
package main

type Svc struct{}

func (s *Svc) Start() {}
"""
    return "go", src, [("method", "Start", "Svc")]


@case(tags=["symbol"])
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
def case_go_type_alias() -> SymbolCase:
    """type ID string."""
    src = """\
package main

type ID string
"""
    return "go", src, [("class", "ID", "")]


@case(tags=["symbol"])
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
def case_go_import_single() -> ImportCase:
    """import "fmt"."""
    src = """\
package main
import "fmt"
"""
    return "go", src, {"module": "fmt"}


@case(tags=["import"])
def case_go_import_nested() -> ImportCase:
    """import "os/exec"."""
    src = """\
package main
import "os/exec"
"""
    return "go", src, {"module": "os/exec"}


@case(tags=["import"])
def case_go_import_url() -> ImportCase:
    """import "github.com/user/repo"."""
    src = """\
package main
import "github.com/user/repo"
"""
    return "go", src, {"module": "github.com/user/repo"}


@case(tags=["import"])
def case_go_import_aliased() -> ImportCase:
    """import f "fmt" — alias ignored, module captured."""
    src = """\
package main
import f "fmt"
"""
    return "go", src, {"module": "fmt"}


@case(tags=["multi_import"])
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
    return "go", src, {"import", "class", "method", "function"}, [("String", "Config")]


# ═════════════════════════════════════════════════════════════════════
# Rust
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
def case_rust_function() -> SymbolCase:
    """fn hello() {}."""
    return "rust", "fn hello() {}\n", [("function", "hello", "")]


@case(tags=["symbol"])
def case_rust_function_params() -> SymbolCase:
    """fn add(a: i32, b: i32) -> i32."""
    return "rust", "fn add(a: i32, b: i32) -> i32 { a + b }\n", [("function", "add", "")]


@case(tags=["symbol"])
def case_rust_pub_function() -> SymbolCase:
    """pub fn visible() {}."""
    return "rust", "pub fn visible() {}\n", [("function", "visible", "")]


@case(tags=["symbol"])
def case_rust_async_function() -> SymbolCase:
    """async fn fetch() {}."""
    return "rust", "async fn fetch() {}\n", [("function", "fetch", "")]


@case(tags=["symbol"])
def case_rust_multiple_functions() -> SymbolCase:
    """Multiple functions."""
    src = """\
fn a() {}
fn b() {}
fn c() {}
"""
    return "rust", src, [("function", "a", ""), ("function", "b", ""), ("function", "c", "")]


@case(tags=["symbol"])
def case_rust_struct() -> SymbolCase:
    """struct User { name: String }."""
    src = """\
struct User {
    name: String,
}
"""
    return "rust", src, [("class", "User", "")]


@case(tags=["symbol"])
def case_rust_unit_struct() -> SymbolCase:
    """struct Marker;"""
    return "rust", "struct Marker;\n", [("class", "Marker", "")]


@case(tags=["symbol"])
def case_rust_tuple_struct() -> SymbolCase:
    """struct Point(f64, f64);"""
    return "rust", "struct Point(f64, f64);\n", [("class", "Point", "")]


@case(tags=["symbol"])
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


@case(tags=["symbol"])
def case_rust_method_named_like_type() -> SymbolCase:
    """An impl method whose name equals its type is addressed `Node::`."""
    src = """\
struct Node {}
impl Node {
    fn Node(&self) {}
}
"""
    return "rust", src, [("method", "Node", "Node")]


@case(tags=["symbol"])
def case_rust_fn_in_mod() -> SymbolCase:
    """A function in a module is addressed by the module.

    Rust `mod` is not tracked today (`bar` → ""); target "outer".
    """
    src = """\
mod outer {
    fn bar() {}
}
"""
    return "rust", src, [("function", "bar", "outer")]


@case(tags=["symbol"])
def case_rust_impl_in_mod() -> SymbolCase:
    """An impl method inside a module carries module::type."""
    src = """\
mod m {
    struct S {}
    impl S {
        fn go(&self) {}
    }
}
"""
    return "rust", src, [("method", "go", "m::S")]


@case(tags=["symbol"])
def case_rust_nested_mod() -> SymbolCase:
    """Nested modules compose."""
    src = """\
mod a {
    mod b {
        fn f() {}
    }
}
"""
    return "rust", src, [("function", "f", "a::b")]


@case(tags=["symbol"])
def case_rust_nested_mod_impl_method() -> SymbolCase:
    """A method in an impl, nested in two modules, composes `a::b::S`."""
    src = """\
mod a {
    mod b {
        struct S {}
        impl S {
            fn go(&self) {}
        }
    }
}
"""
    return "rust", src, [("method", "go", "a::b::S")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
def case_rust_import_scoped() -> ImportCase:
    """use std::collections::HashMap."""
    return (
        "rust",
        "use std::collections::HashMap;\n",
        {"module": "std/collections", "names": "HashMap"},
    )


@case(tags=["import"])
def case_rust_import_deeply_nested() -> ImportCase:
    """use a::b::c::d::Item."""
    return "rust", "use a::b::c::d::Item;\n", {"module": "a/b/c/d", "names": "Item"}


@case(tags=["import"])
def case_rust_import_crate() -> ImportCase:
    """use crate::models::Chunk."""
    return "rust", "use crate::models::Chunk;\n", {"module": "crate/models", "names": "Chunk"}


@case(tags=["import"])
def case_rust_import_crate_braces() -> ImportCase:
    """use crate::models::{Chunk, Edge}."""
    return (
        "rust",
        "use crate::models::{Chunk, Edge};\n",
        {"module": "crate/models", "names": "Chunk,Edge"},
    )


@case(tags=["import"])
def case_rust_import_super_single() -> ImportCase:
    """use super::utils."""
    return "rust", "use super::utils;\n", {"names": "utils", "dots": "2"}


@case(tags=["import"])
def case_rust_import_super_nested() -> ImportCase:
    """use super::helpers::run."""
    return (
        "rust",
        "use super::helpers::run;\n",
        {"module": "helpers", "names": "run", "dots": "2"},
    )


@case(tags=["import"])
def case_rust_import_super_super() -> ImportCase:
    """use super::super::common::Config — double super."""
    return (
        "rust",
        "use super::super::common::Config;\n",
        {"dots": "3", "module": "common", "names": "Config"},
    )


@case(tags=["import"])
def case_rust_import_use_list() -> ImportCase:
    """use std::io::{Read, Write}."""
    return "rust", "use std::io::{Read, Write};\n", {"module": "std/io", "names": "Read,Write"}


@case(tags=["import"])
def case_rust_import_use_list_self() -> ImportCase:
    """use std::io::{self, Read}."""
    return "rust", "use std::io::{self, Read};\n", {"module": "std/io", "names": "self,Read"}


@case(tags=["import"])
def case_rust_import_bare() -> ImportCase:
    """use serde;"""
    return "rust", "use serde;\n", {"module": "serde"}


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
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
def case_java_class() -> SymbolCase:
    """class User {}."""
    return "java", "class User {}\n", [("class", "User", "")]


@case(tags=["symbol"])
def case_java_public_class() -> SymbolCase:
    """public class App {}."""
    return "java", "public class App {}\n", [("class", "App", "")]


@case(tags=["symbol"])
def case_java_class_extends() -> SymbolCase:
    """class Admin extends User {}."""
    return "java", "class Admin extends User {}\n", [("class", "Admin", "")]


@case(tags=["symbol"])
def case_java_class_implements() -> SymbolCase:
    """class UserService implements Service {}."""
    return "java", "class UserService implements Service {}\n", [("class", "UserService", "")]


@case(tags=["symbol"])
def case_java_multiple_classes() -> SymbolCase:
    """Multiple classes."""
    src = """\
class Foo {}
class Bar {}
"""
    return "java", src, [("class", "Foo", ""), ("class", "Bar", "")]


@case(tags=["symbol"])
def case_java_method_in_class() -> SymbolCase:
    """Method scoped to class."""
    src = """\
class Service {
    void process() {}
}
"""
    return "java", src, [("method", "process", "Service")]


@case(tags=["symbol"])
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
def case_java_static_method() -> SymbolCase:
    """Static method still scoped."""
    src = """\
class Factory {
    static Object create() { return null; }
}
"""
    return "java", src, [("method", "create", "Factory")]


@case(tags=["symbol"])
def case_java_method_params() -> SymbolCase:
    """Method with parameters."""
    src = """\
class Calc {
    int add(int a, int b) { return a + b; }
}
"""
    return "java", src, [("method", "add", "Calc")]


@case(tags=["symbol"])
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
        [("class", "Outer", ""), ("class", "Inner", "Outer"), ("method", "deep", "Outer::Inner")],
    )


@case(tags=["symbol"])
def case_java_triple_nested_class() -> SymbolCase:
    """Three levels of nested class compose the full path."""
    src = """\
class Outer {
    class Mid {
        class Inner {
            void deep() {}
        }
    }
}
"""
    return "java", src, [("method", "deep", "Outer::Mid::Inner")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
def case_java_import_class() -> ImportCase:
    """import java.util.HashMap."""
    return "java", "import java.util.HashMap;\n", {"module": "java.util.HashMap"}


@case(tags=["import"])
def case_java_import_deeply_nested() -> ImportCase:
    """import com.example.app.models.User."""
    return (
        "java",
        "import com.example.app.models.User;\n",
        {"module": "com.example.app.models.User"},
    )


@case(tags=["import"])
def case_java_import_static() -> ImportCase:
    """import static org.junit.Assert.assertEquals."""
    return (
        "java",
        "import static org.junit.Assert.assertEquals;\n",
        {"module": "org.junit.Assert.assertEquals"},
    )


@case(tags=["import"])
def case_java_import_static_method() -> ImportCase:
    """import static java.util.Collections.sort."""
    return (
        "java",
        "import static java.util.Collections.sort;\n",
        {"module": "java.util.Collections.sort"},
    )


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
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
def case_c_function_basic() -> SymbolCase:
    """int add(int a, int b)."""
    return "c", "int add(int a, int b) { return a + b; }\n", [("function", "add", "")]


@case(tags=["symbol"])
def case_c_function_void() -> SymbolCase:
    """void do_stuff(void)."""
    return "c", "void do_stuff(void) { }\n", [("function", "do_stuff", "")]


@case(tags=["symbol"])
def case_c_function_static() -> SymbolCase:
    """static int helper(void)."""
    return "c", "static int helper(void) { return 1; }\n", [("function", "helper", "")]


@case(tags=["symbol"])
def case_c_multiple_functions() -> SymbolCase:
    """Multiple C functions."""
    src = """\
int foo(void) { return 0; }
void bar(void) { }
"""
    return "c", src, [("function", "foo", ""), ("function", "bar", "")]


@case(tags=["symbol"])
def case_c_struct() -> SymbolCase:
    """struct Node."""
    return "c", "struct Node { int value; };\n", [("class", "Node", "")]


@case(tags=["symbol"])
def case_c_enum() -> SymbolCase:
    """enum Color."""
    return "c", "enum Color { RED, GREEN, BLUE };\n", [("class", "Color", "")]


@case(tags=["symbol"])
def case_c_typedef_struct() -> SymbolCase:
    """typedef struct { ... } Point."""
    return "c", "typedef struct { int x; int y; } Point;\n", [("class", "Point", "")]


@case(tags=["symbol"])
def case_c_no_scope() -> SymbolCase:
    """C functions are never scoped — no classes."""
    src = """\
struct S { int x; };
int func(void) { return 0; }
"""
    return "c", src, [("function", "func", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
def case_c_include_system() -> ImportCase:
    """#include <stdio.h>."""
    return "c", "#include <stdio.h>\n", {"module": "stdio.h"}


@case(tags=["import"])
def case_c_include_local() -> ImportCase:
    """#include "mylib.h"."""
    return "c", '#include "mylib.h"\n', {"module": "mylib.h"}


@case(tags=["import"])
def case_c_include_nested_path() -> ImportCase:
    """#include "utils/helpers.h"."""
    return "c", '#include "utils/helpers.h"\n', {"module": "utils/helpers.h"}


@case(tags=["import"])
def case_c_include_system_nested() -> ImportCase:
    """#include <sys/types.h>."""
    return "c", "#include <sys/types.h>\n", {"module": "sys/types.h"}


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
def case_c_multiple_includes() -> MultiImportCase:
    """Two include directives."""
    src = """\
#include <stdlib.h>
#include "local.h"
"""
    return "c", src, 2, [{"module": "stdlib.h"}, {"module": "local.h"}]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
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
def case_cpp_free_function_void() -> SymbolCase:
    """void greet()."""
    return "cpp", "void greet() { }\n", [("function", "greet", "")]


@case(tags=["symbol"])
def case_cpp_free_function_returning() -> SymbolCase:
    """int compute(int x)."""
    return "cpp", "int compute(int x) { return x; }\n", [("function", "compute", "")]


@case(tags=["symbol"])
def case_cpp_multiple_functions() -> SymbolCase:
    """Multiple C++ functions."""
    src = """\
int foo() { return 0; }
void bar() { }
"""
    return "cpp", src, [("function", "foo", ""), ("function", "bar", "")]


@case(tags=["symbol"])
def case_cpp_class() -> SymbolCase:
    """class Shape."""
    return "cpp", "class Shape { };\n", [("class", "Shape", "")]


@case(tags=["symbol"])
def case_cpp_struct() -> SymbolCase:
    """struct Point."""
    return "cpp", "struct Point { double x; double y; };\n", [("class", "Point", "")]


@case(tags=["symbol"])
def case_cpp_enum_class() -> SymbolCase:
    """enum class Color."""
    return "cpp", "enum class Color { Red, Green, Blue };\n", [("class", "Color", "")]


@case(tags=["symbol"])
def case_cpp_class_inheritance() -> SymbolCase:
    """class Derived : public Base."""
    src = """\
class Base { };
class Derived : public Base { };
"""
    return "cpp", src, [("class", "Base", ""), ("class", "Derived", "")]


@case(tags=["symbol"])
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
def case_cpp_struct_method() -> SymbolCase:
    """Method scoped to struct."""
    return "cpp", "struct Vec { void push(int v) { } };\n", [("method", "push", "Vec")]


@case(tags=["symbol"])
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
def case_cpp_namespace_function() -> SymbolCase:
    """A free function in a namespace is addressed by the namespace.

    C++ `namespace` is not tracked today (`f` → ""); target "ns".
    """
    src = """\
namespace ns {
void f() { }
}
"""
    return "cpp", src, [("function", "f", "ns")]


@case(tags=["symbol"])
def case_cpp_namespace_class_method() -> SymbolCase:
    """A method in a class in a namespace carries the full path."""
    src = """\
namespace ns {
class Widget {
public:
    void draw() { }
};
}
"""
    return "cpp", src, [("class", "Widget", "ns"), ("method", "draw", "ns::Widget")]


@case(tags=["symbol"])
def case_cpp_nested_namespace() -> SymbolCase:
    """Nested namespaces compose outermost-first."""
    src = """\
namespace a {
namespace b {
void f() { }
}
}
"""
    return "cpp", src, [("function", "f", "a::b")]


@case(tags=["symbol"])
def case_cpp_namespace_nested_class_method() -> SymbolCase:
    """A method in nested classes, inside a namespace, composes the full path.

    Mixes a namespace with class nesting — `ns::Outer::Inner`.
    """
    src = """\
namespace ns {
class Outer {
public:
    class Inner {
    public:
        void deep() { }
    };
};
}
"""
    return "cpp", src, [("method", "deep", "ns::Outer::Inner")]


@case(tags=["symbol"])
def case_cpp_free_function_not_scoped() -> SymbolCase:
    """Function after class is not scoped."""
    src = """\
class C { };
void standalone() { }
"""
    return "cpp", src, [("function", "standalone", "")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
def case_cpp_include_system() -> ImportCase:
    """#include <iostream>."""
    return "cpp", "#include <iostream>\n", {"module": "iostream"}


@case(tags=["import"])
def case_cpp_include_local() -> ImportCase:
    """#include "myheader.h"."""
    return "cpp", '#include "myheader.h"\n', {"module": "myheader.h"}


@case(tags=["import"])
def case_cpp_include_nested() -> ImportCase:
    """#include <boost/optional.hpp>."""
    return "cpp", "#include <boost/optional.hpp>\n", {"module": "boost/optional.hpp"}


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
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
    return "markdown", src, [("doc_section", "Deep", "Top::Mid")]


@case(tags=["symbol"])
def case_md_same_name_under_different_parents() -> SymbolCase:
    """Same-named sections under different parents get distinct scopes.

    Two `Overview` subsections — one under `A`, one under `B` — are
    `(doc_section, Overview, A)` and `(doc_section, Overview, B)`. Without
    the full heading path they would collide on identity; addressing
    keeps them apart.
    """
    src = """\
# A

Alpha intro.

## Overview

Alpha overview.

# B

Beta intro.

## Overview

Beta overview.
"""
    return (
        "markdown",
        src,
        [("doc_section", "Overview", "A"), ("doc_section", "Overview", "B")],
    )


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
            ("doc_section", "Title", ""),
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
    return "rst", src, [("doc_section", "Deep", "Top::Mid")]


@case(tags=["symbol"])
def case_rst_same_name_under_different_parents() -> SymbolCase:
    """Same-named RST subsections under different parents stay distinct.

    Both `Overview` subsections are scoped by their parent only — `A`
    and `B` — never themselves, so they stay distinct.
    """
    src = """\
A
=

Alpha intro.

Overview
--------

Alpha overview.

B
=

Beta intro.

Overview
--------

Beta overview.
"""
    return (
        "rst",
        src,
        [("doc_section", "Overview", "A"), ("doc_section", "Overview", "B")],
    )


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


# ══════════════════════════════════════════════════��══════════════════
# TOML
# ════════════��════════════════════════════════════════════════════════


@case(tags=["symbol"])
def case_toml_splits_by_table() -> SymbolCase:
    """TOML splits by tables; a dotted table is named by its last segment."""
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


# ═════════════���═══════════════════════════════════════════════════════
# HTML
# ════════════════════════════════════════════════════════════════════��


@case(tags=["symbol"])
def case_html_semantic_elements() -> SymbolCase:
    """HTML names an element by its id (even beside a class), else its tag."""
    src = """\
<html>
<body>
  <main id="content"><p>Hello</p></main>
  <article class="post" id="first">Text</article>
  <nav>Links</nav>
</body>
</html>
"""
    return (
        "html",
        src,
        [
            ("doc_section", "body", ""),
            ("doc_section", "content", ""),
            ("doc_section", "first", ""),
            ("doc_section", "nav", ""),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# SQL
# ═════════════════════════════════════════════════════════════════════
#
# The cases below are the spec of the SQL constructs we support. One
# chunk is produced per top-level statement (plus one per CTE), named by
# its object/table; SQL has no nesting so every scope is "". Construct
# cases pin (kind, name); invariance cases prove optional clauses don't
# change the name; edge cases cover anonymous statements, error recovery,
# and the known grammar gaps. SQL keywords are uppercase by convention.

# ── DDL definitions: class ───────────────────────────────────────────


@case(tags=["symbol"])
def case_sql_table() -> SymbolCase:
    """CREATE TABLE — structural definition → class."""
    return "sql", "CREATE TABLE users (id INT, name TEXT);\n", [("class", "users", "")]


@case(tags=["symbol"])
def case_sql_view() -> SymbolCase:
    """CREATE VIEW → class."""
    return "sql", "CREATE VIEW active AS SELECT * FROM users;\n", [("class", "active", "")]


@case(tags=["symbol"])
def case_sql_materialized_view() -> SymbolCase:
    """CREATE MATERIALIZED VIEW → class."""
    src = "CREATE MATERIALIZED VIEW recent AS SELECT * FROM users;\n"
    return "sql", src, [("class", "recent", "")]


@case(tags=["symbol"])
def case_sql_type_enum() -> SymbolCase:
    """CREATE TYPE ... AS ENUM — SQL's enum → class."""
    return "sql", "CREATE TYPE mood AS ENUM ('sad', 'happy');\n", [("class", "mood", "")]


@case(tags=["symbol"])
def case_sql_type_composite() -> SymbolCase:
    """CREATE TYPE ... AS (...) — composite type → class."""
    src = "CREATE TYPE point AS (x DOUBLE PRECISION, y DOUBLE PRECISION);\n"
    return "sql", src, [("class", "point", "")]


# ── DDL definitions: variable (standalone named objects) ─────────────


@case(tags=["symbol"])
def case_sql_index() -> SymbolCase:
    """CREATE INDEX → variable, named by the index (not the ON table)."""
    return "sql", "CREATE INDEX idx_name ON users (name);\n", [("variable", "idx_name", "")]


@case(tags=["symbol"])
def case_sql_sequence() -> SymbolCase:
    """CREATE SEQUENCE → variable."""
    return "sql", "CREATE SEQUENCE order_id START 1;\n", [("variable", "order_id", "")]


@case(tags=["symbol"])
def case_sql_schema() -> SymbolCase:
    """CREATE SCHEMA → variable."""
    return "sql", "CREATE SCHEMA app;\n", [("variable", "app", "")]


@case(tags=["symbol"])
def case_sql_extension() -> SymbolCase:
    """CREATE EXTENSION → variable (common in migrations)."""
    return "sql", "CREATE EXTENSION postgis;\n", [("variable", "postgis", "")]


@case(tags=["symbol"])
def case_sql_trigger() -> SymbolCase:
    """CREATE TRIGGER → variable, named by the trigger (not its table/function)."""
    src = """\
CREATE TRIGGER audit BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION log_change();
"""
    return "sql", src, [("variable", "audit", "")]


# ── Routines and statements: function ────────────────────────────────


@case(tags=["symbol"])
def case_sql_function() -> SymbolCase:
    """CREATE FUNCTION — routine → function."""
    src = """\
CREATE FUNCTION add(a INT, b INT) RETURNS INT
LANGUAGE SQL
AS $$ SELECT a + b; $$;
"""
    return "sql", src, [("function", "add", "")]


@case(tags=["symbol"])
def case_sql_select() -> SymbolCase:
    """SELECT → function, named by its primary FROM table."""
    return "sql", "SELECT id, name FROM users;\n", [("function", "users", "")]


@case(tags=["symbol"])
def case_sql_insert() -> SymbolCase:
    """INSERT → function, named by its target table."""
    return "sql", "INSERT INTO logs (msg) VALUES ('hi');\n", [("function", "logs", "")]


@case(tags=["symbol"])
def case_sql_update() -> SymbolCase:
    """UPDATE → function, named by its target table."""
    return "sql", "UPDATE users SET name = 'x' WHERE id = 1;\n", [("function", "users", "")]


@case(tags=["symbol"])
def case_sql_delete() -> SymbolCase:
    """DELETE → function, named by its target table."""
    return "sql", "DELETE FROM sessions WHERE id = 1;\n", [("function", "sessions", "")]


@case(tags=["symbol"])
def case_sql_cte() -> SymbolCase:
    """Each CTE → a function named by its identifier."""
    src = """\
WITH ranked AS (
    SELECT id FROM events
),
recent AS (
    SELECT id FROM ranked
)
SELECT id FROM recent;
"""
    return "sql", src, [("function", "ranked", ""), ("function", "recent", "")]


# ── DDL operations: function (alter/drop) ────────────────────────────


@case(tags=["symbol"])
def case_sql_alter_table() -> SymbolCase:
    """ALTER TABLE → function, named by the table."""
    return "sql", "ALTER TABLE users ADD COLUMN age INT;\n", [("function", "users", "")]


@case(tags=["symbol"])
def case_sql_drop_table() -> SymbolCase:
    """DROP TABLE → function, named by the table."""
    return "sql", "DROP TABLE legacy;\n", [("function", "legacy", "")]


@case(tags=["symbol"])
def case_sql_drop_index() -> SymbolCase:
    """DROP INDEX → function, named by the index."""
    return "sql", "DROP INDEX idx_old;\n", [("function", "idx_old", "")]


# ── Naming invariances (optional clauses must not change the name) ───


@case(tags=["symbol"])
def case_sql_table_if_not_exists() -> SymbolCase:
    """IF NOT EXISTS does not change the table name."""
    return "sql", "CREATE TABLE IF NOT EXISTS users (id INT);\n", [("class", "users", "")]


@case(tags=["symbol"])
def case_sql_view_or_replace() -> SymbolCase:
    """CREATE OR REPLACE does not change the view name."""
    return "sql", "CREATE OR REPLACE VIEW v AS SELECT 1;\n", [("class", "v", "")]


@case(tags=["symbol"])
def case_sql_schema_qualified_table() -> SymbolCase:
    """A schema-qualified table is named by the table, not the schema."""
    return "sql", "CREATE TABLE app.users (id INT);\n", [("class", "users", "")]


@case(tags=["symbol"])
def case_sql_select_aliased_table() -> SymbolCase:
    """An aliased FROM is named by the table, not the alias."""
    src = "SELECT c.id FROM chunks AS c JOIN files f ON f.id = c.id;\n"
    return "sql", src, [("function", "chunks", "")]


@case(tags=["symbol"])
def case_sql_create_table_as_select() -> SymbolCase:
    """CREATE TABLE AS SELECT yields only the table, not the nested select."""
    return "sql", "CREATE TABLE snap AS SELECT * FROM users;\n", [("class", "snap", "")]


@case(tags=["symbol"])
def case_sql_insert_select() -> SymbolCase:
    """INSERT ... SELECT is named by the insert target."""
    src = "INSERT INTO archive SELECT * FROM events;\n"
    return "sql", src, [("function", "archive", "")]


# ── Anonymous statements (no nameable target) ────────────────────────


@case(tags=["symbol"])
def case_sql_select_no_table() -> SymbolCase:
    """A SELECT with no table is anonymous."""
    return "sql", "SELECT 1;\n", [("function", "<anonymous>", "")]


@case(tags=["symbol"])
def case_sql_union() -> SymbolCase:
    """A top-level UNION (set_operation) is one anonymous function chunk."""
    src = "SELECT id FROM a UNION SELECT id FROM b;\n"
    return "sql", src, [("function", "<anonymous>", "")]


@case(tags=["symbol"])
def case_sql_param_error_recovery() -> SymbolCase:
    """A DuckDB $param errors internally but the statement still extracts.

    The `$id` placeholder is unknown to the grammar (an ERROR node),
    but error recovery keeps the enclosing statement, so the chunk is
    still produced and named by its table.
    """
    return "sql", "SELECT * FROM users WHERE id = $id;\n", [("function", "users", "")]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
def case_sql_migration() -> MixedCase:
    """Realistic migration script producing every SQL chunk kind.

    SQL has no method scoping, so the expected-methods list is
    empty — definitions are all top-level.
    """
    src = """\
CREATE SCHEMA shop;

CREATE TABLE shop.products (
    id INT PRIMARY KEY,
    name TEXT NOT NULL,
    price NUMERIC
);

CREATE VIEW in_stock AS
SELECT * FROM shop.products WHERE price > 0;

CREATE FUNCTION price_with_tax(p NUMERIC) RETURNS NUMERIC
LANGUAGE SQL
AS $$ SELECT p * 1.2; $$;

CREATE INDEX idx_products_name ON shop.products (name);
"""
    return (
        "sql",
        src,
        {"class", "function", "variable"},
        [],
    )


# ═════════════════════════════════════════════════════════════════════
# Module-level variables (cross-language fan-out)
# ═════════════════════════════════════════════════════════════════════


@case(tags=["symbol"])
def case_go_package_var() -> SymbolCase:
    """Package-level var."""
    return "go", "package main\nvar MaxSize = 100\n", [("variable", "MaxSize", "")]


@case(tags=["symbol"])
def case_go_package_const() -> SymbolCase:
    """Package-level const."""
    return "go", "package main\nconst Timeout = 30\n", [("variable", "Timeout", "")]


@case(tags=["symbol"])
def case_rust_const() -> SymbolCase:
    """Crate-level const."""
    return "rust", "const MAX: i32 = 100;\n", [("variable", "MAX", "")]


@case(tags=["symbol"])
def case_rust_static() -> SymbolCase:
    """Crate-level static."""
    return "rust", 'static NAME: &str = "x";\n', [("variable", "NAME", "")]


@case(tags=["symbol"])
def case_c_global() -> SymbolCase:
    """File-scope global with initialiser."""
    return "c", "int g = 5;\n", [("variable", "g", "")]


@case(tags=["symbol"])
def case_cpp_global() -> SymbolCase:
    """File-scope global with initialiser."""
    return "cpp", "int g = 5;\n", [("variable", "g", "")]


# ═════════════════════════════════════════════════════════════════════
# Module-level destructuring & multiple assignment (flat)
# ═════════════════════════════════════════════════════════════════════

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_py_tuple_unpack() -> SymbolCase:
    """Flat tuple unpacking."""
    return "python", "a, b = compute()\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_py_paren_tuple_unpack() -> SymbolCase:
    """Parenthesised tuple target."""
    return "python", "(a, b) = compute()\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_py_list_unpack() -> SymbolCase:
    """List-pattern target."""
    return "python", "[a, b] = compute()\n", [("variable", "a", ""), ("variable", "b", "")]


@case(tags=["symbol"])
def case_py_star_unpack() -> SymbolCase:
    """Starred target."""
    return "python", "a, *rest = compute()\n", [("variable", "a", ""), ("variable", "rest", "")]


@case(tags=["symbol"])
def case_go_grouped_var() -> SymbolCase:
    """Go grouped var block."""
    src = """\
package m

var (
	X = 1
	Y = 2
)
"""
    return "go", src, [("variable", "X", ""), ("variable", "Y", "")]


@case(tags=["symbol"])
def case_go_grouped_const() -> SymbolCase:
    """Go grouped const block (already supported — regression guard)."""
    src = """\
package m

const (
	A = 1
	B = 2
)
"""
    return "go", src, [("variable", "A", ""), ("variable", "B", "")]


# ── Known limitations: nested / chained (strict xfail) ───────────────


@case(tags=["symbol"], marks=_xfail_nested)
def case_py_nested_unpack_xfail() -> SymbolCase:
    """Nested tuple unpacking — only the outer level is captured today."""
    return (
        "python",
        "(a, b), c = f()\n",
        [("variable", "a", ""), ("variable", "b", ""), ("variable", "c", "")],
    )


@case(tags=["symbol"], marks=_xfail_nested)
def case_py_chained_assignment_xfail() -> SymbolCase:
    """Chained assignment — only the first target is captured today."""
    return "python", "a = b = f()\n", [("variable", "a", ""), ("variable", "b", "")]
