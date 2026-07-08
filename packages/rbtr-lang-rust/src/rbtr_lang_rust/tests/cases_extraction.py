"""Rust extraction test cases."""

from __future__ import annotations

import pytest
from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


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


@case(tags=["symbol"])
def case_rust_const() -> SymbolCase:
    """Crate-level const."""
    return "rust", "const MAX: i32 = 100;\n", [("variable", "MAX", "")]


@case(tags=["symbol"])
def case_rust_static() -> SymbolCase:
    """Crate-level static."""
    return "rust", 'static NAME: &str = "x";\n', [("variable", "NAME", "")]
