"""Rust extraction test cases."""

from __future__ import annotations

import pytest
from pytest_cases import case

from rbtr.testing.assertions import Referrer, Target

type SymbolCase = tuple[dict[str, str], list[tuple[str, str, str]]]
type ImportCase = tuple[dict[str, str], dict[str, str | list[dict[str, str]]]]
type MixedCase = tuple[dict[str, str], set[str], list[tuple[str, str]]]

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_rust_function() -> SymbolCase:
    """fn hello() {}."""
    return {"input.rs": "fn hello() {}\n"}, [("function", "hello", "")]


@case(tags=["symbol"])
def case_rust_function_params() -> SymbolCase:
    """fn add(a: i32, b: i32) -> i32."""
    return {"input.rs": "fn add(a: i32, b: i32) -> i32 { a + b }\n"}, [("function", "add", "")]


@case(tags=["symbol"])
def case_rust_pub_function() -> SymbolCase:
    """pub fn visible() {}."""
    return {"input.rs": "pub fn visible() {}\n"}, [("function", "visible", "")]


@case(tags=["symbol"])
def case_rust_async_function() -> SymbolCase:
    """async fn fetch() {}."""
    return {"input.rs": "async fn fetch() {}\n"}, [("function", "fetch", "")]


@case(tags=["symbol"])
def case_rust_multiple_functions() -> SymbolCase:
    """Multiple functions."""
    src = """\
fn a() {}
fn b() {}
fn c() {}
"""
    return {"input.rs": src}, [("function", "a", ""), ("function", "b", ""), ("function", "c", "")]


@case(tags=["symbol"])
def case_rust_struct() -> SymbolCase:
    """struct User { name: String }."""
    src = """\
struct User {
    name: String,
}
"""
    return {"input.rs": src}, [("class", "User", "")]


@case(tags=["symbol"])
def case_rust_unit_struct() -> SymbolCase:
    """struct Marker;"""
    return {"input.rs": "struct Marker;\n"}, [("class", "Marker", "")]


@case(tags=["symbol"])
def case_rust_tuple_struct() -> SymbolCase:
    """struct Point(f64, f64);"""
    return {"input.rs": "struct Point(f64, f64);\n"}, [("class", "Point", "")]


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
    return {"input.rs": src}, [("class", "Color", "")]


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
    return {"input.rs": src}, [("method", "start", "Svc"), ("method", "stop", "Svc")]


@case(tags=["symbol"])
def case_rust_method_named_like_type() -> SymbolCase:
    """An impl method whose name equals its type is addressed `Node::`."""
    src = """\
struct Node {}
impl Node {
    fn Node(&self) {}
}
"""
    return {"input.rs": src}, [("method", "Node", "Node")]


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
    return {"input.rs": src}, [("function", "bar", "outer")]


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
    return {"input.rs": src}, [("method", "go", "m::S")]


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
    return {"input.rs": src}, [("function", "f", "a::b")]


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
    return {"input.rs": src}, [("method", "go", "a::b::S")]


@case(tags=["import"])
def case_rust_import_scoped() -> ImportCase:
    """use std::collections::HashMap."""
    return (
        {"input.rs": "use std::collections::HashMap;\n"},
        {
            "module": "std::collections",
            "names": "HashMap",
            "bindings": [{"name": "HashMap", "local": "HashMap"}],
        },
    )


@case(tags=["import"])
def case_rust_import_deeply_nested() -> ImportCase:
    """use a::b::c::d::Item."""
    return (
        {"input.rs": "use a::b::c::d::Item;\n"},
        {"module": "a::b::c::d", "names": "Item", "bindings": [{"name": "Item", "local": "Item"}]},
    )


@case(tags=["import"])
def case_rust_import_crate() -> ImportCase:
    """use crate::models::Chunk."""
    return (
        {"input.rs": "use crate::models::Chunk;\n"},
        {
            "module": "crate::models",
            "names": "Chunk",
            "bindings": [{"name": "Chunk", "local": "Chunk"}],
        },
    )


@case(tags=["import"])
def case_rust_import_crate_braces() -> ImportCase:
    """use crate::models::{Chunk, Edge}."""
    return (
        {"input.rs": "use crate::models::{Chunk, Edge};\n"},
        {
            "module": "crate::models",
            "names": "Chunk,Edge",
            "bindings": [{"name": "Chunk", "local": "Chunk"}, {"name": "Edge", "local": "Edge"}],
        },
    )


@case(tags=["import"])
def case_rust_mod_declaration() -> ImportCase:
    """mod util; — a child of this module, bound under its name."""
    return (
        {"input.rs": "mod util;\n"},
        {
            "module": "self::util",
            "names": "",
            "bindings": [{"name": "self::util", "local": "util"}],
        },
    )


@case(tags=["import"])
def case_rust_import_super_single() -> ImportCase:
    """use super::utils."""
    return (
        {"input.rs": "use super::utils;\n"},
        {"module": "super", "names": "utils", "bindings": [{"name": "utils", "local": "utils"}]},
    )


@case(tags=["import"])
def case_rust_import_super_nested() -> ImportCase:
    """use super::helpers::run."""
    return (
        {"input.rs": "use super::helpers::run;\n"},
        {"module": "super::helpers", "names": "run", "bindings": [{"name": "run", "local": "run"}]},
    )


@case(tags=["import"])
def case_rust_import_super_super() -> ImportCase:
    """use super::super::common::Config — double super."""
    return (
        {"input.rs": "use super::super::common::Config;\n"},
        {
            "module": "super::super::common",
            "names": "Config",
            "bindings": [{"name": "Config", "local": "Config"}],
        },
    )


@case(tags=["import"])
def case_rust_import_use_list() -> ImportCase:
    """use std::io::{Read, Write}."""
    return (
        {"input.rs": "use std::io::{Read, Write};\n"},
        {
            "module": "std::io",
            "names": "Read,Write",
            "bindings": [{"name": "Read", "local": "Read"}, {"name": "Write", "local": "Write"}],
        },
    )


@case(tags=["import"])
def case_rust_import_use_list_self() -> ImportCase:
    """use std::io::{self, Read}."""
    return (
        {"input.rs": "use std::io::{self, Read};\n"},
        {
            "module": "std::io",
            "names": "self,Read",
            "bindings": [{"name": "std::io", "local": "io"}, {"name": "Read", "local": "Read"}],
        },
    )


@case(tags=["import"])
def case_rust_import_bare() -> ImportCase:
    """use serde;"""
    return (
        {"input.rs": "use serde;\n"},
        {"module": "serde", "bindings": [{"name": "serde", "local": "serde"}]},
    )


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
    return (
        {"input.rs": src},
        {"import", "class", "method", "function"},
        [("new", "App"), ("run", "App")],
    )


@case(tags=["symbol"])
def case_rust_const() -> SymbolCase:
    """Crate-level const."""
    return {"input.rs": "const MAX: i32 = 100;\n"}, [("variable", "MAX", "")]


@case(tags=["symbol"])
def case_rust_static() -> SymbolCase:
    """Crate-level static."""
    return {"input.rs": 'static NAME: &str = "x";\n'}, [("variable", "NAME", "")]


@case(tags=["import"])
def case_rust_import_aliased() -> ImportCase:
    """use a::b as c; — `b` bound under `c`."""
    return (
        {"input.rs": "use a::b as c;\n"},
        {"module": "a", "names": "b", "bindings": [{"name": "b", "local": "c"}]},
    )


@case(tags=["import"])
def case_rust_import_aliased_in_list() -> ImportCase:
    """use std::io::{Read as R}; — `Read` bound under `R`."""
    return (
        {"input.rs": "use std::io::{Read as R};\n"},
        {"module": "std::io", "names": "Read", "bindings": [{"name": "Read", "local": "R"}]},
    )


@case(tags=["import"])
def case_rust_import_glob() -> ImportCase:
    """use crate::models::*; — every public name, written `*`."""
    return {"input.rs": "use crate::models::*;\n"}, {"module": "crate::models", "names": "*"}


type ReferencesCase = tuple[dict[str, str], dict[Target, set[Referrer]]]


@case(tags=["references"])
def case_rust_calls() -> ReferencesCase:
    """Each call shape; `self` and `Self` reach the impl's type, a macro is no call.

    `z` is a function of its own, which `x.y().z()`, a call through a value,
    does not reach, and `println` one which `println!` does not.
    """
    order = """\
use crate::models;

pub struct Order;

fn helper() {}

fn foo<T>() {}

fn z() {}

fn println() {}

impl Order {
    fn new() -> Self {
        Order
    }

    fn check(&self) {}

    fn save(&self) {
        self.check();
        helper();
        Self::new();
        Order::new();
        models::make();
        foo::<u8>();
        x.y().z();
        println!("x");
    }
}
"""
    manifest = """\
[package]
name = "shop"
"""
    lib = """\
mod models;
mod order;
"""
    return (
        {
            "Cargo.toml": manifest,
            "src/lib.rs": lib,
            "src/models.rs": "pub fn make() {}\n",
            "src/order.rs": order,
        },
        {
            ("src/order.rs", "check"): {("src/order.rs", "save", "calls")},
            ("src/order.rs", "helper"): {("src/order.rs", "save", "calls")},
            ("src/order.rs", "new"): {("src/order.rs", "save", "calls")},
            ("src/models.rs", "make"): {("src/order.rs", "save", "calls")},
            ("src/order.rs", "foo"): {("src/order.rs", "save", "calls")},
            ("src/order.rs", "z"): set(),
            ("src/order.rs", "println"): set(),
        },
    )
