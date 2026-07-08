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


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════
# Go
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


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Multi-import ─────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ── Symbols ──────────────────────────────────────────────────────────


# ── Imports ──────────────────────────────────────────────────────────


# ── Mixed ────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════
# Ruby
# ═════════════════════════════════════════════════════════════════════

# ── Symbols ──────────────────────────────────────────────────────────


@case(tags=["symbol"])
def case_ruby_method_no_args() -> SymbolCase:
    """def greet — top-level function."""
    src = """\
def greet
  puts "hello"
end
"""
    return "ruby", src, [("function", "greet", "")]


@case(tags=["symbol"])
def case_ruby_method_with_args() -> SymbolCase:
    """def add(a, b) — top-level function."""
    src = """\
def add(a, b)
  a + b
end
"""
    return "ruby", src, [("function", "add", "")]


@case(tags=["symbol"])
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
def case_ruby_class() -> SymbolCase:
    """class Shape."""
    src = """\
class Shape
end
"""
    return "ruby", src, [("class", "Shape", "")]


@case(tags=["symbol"])
def case_ruby_module() -> SymbolCase:
    """module Utils."""
    src = """\
module Utils
end
"""
    return "ruby", src, [("class", "Utils", "")]


@case(tags=["symbol"])
def case_ruby_class_superclass() -> SymbolCase:
    """class Circle < Shape."""
    src = """\
class Circle < Shape
end
"""
    return "ruby", src, [("class", "Circle", "")]


@case(tags=["symbol"])
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
        [
            ("class", "Utils", ""),
            ("class", "Parser", "Utils"),
            ("method", "parse", "Utils::Parser"),
        ],
    )


@case(tags=["symbol"])
def case_ruby_module_in_module() -> SymbolCase:
    """A method in a module nested in a module carries the full path."""
    src = """\
module A
  module B
    def f
      1
    end
  end
end
"""
    return "ruby", src, [("method", "f", "A::B")]


@case(tags=["symbol"])
def case_ruby_module_module_class_method() -> SymbolCase:
    """A method nested module::module::class carries the full path."""
    src = """\
module A
  module B
    class C
      def go
        1
      end
    end
  end
end
"""
    return "ruby", src, [("method", "go", "A::B::C")]


# ── Imports ──────────────────────────────────────────────────────────


@case(tags=["import"])
def case_ruby_require_simple() -> ImportCase:
    """require "json"."""
    return "ruby", 'require "json"\n', {"module": "json"}


@case(tags=["import"])
def case_ruby_require_nested() -> ImportCase:
    """require "net/http"."""
    return "ruby", 'require "net/http"\n', {"module": "net/http"}


@case(tags=["import"])
def case_ruby_require_relative() -> ImportCase:
    """require_relative "helpers"."""
    return "ruby", 'require_relative "helpers"\n', {"module": "helpers", "dots": "1"}


@case(tags=["import"])
def case_ruby_require_relative_nested() -> ImportCase:
    """require_relative "lib/utils"."""
    return "ruby", 'require_relative "lib/utils"\n', {"module": "lib/utils", "dots": "1"}


@case(tags=["import"])
def case_ruby_require_relative_dot_prefix() -> ImportCase:
    """require_relative "./config" — the `./` is stripped into dots."""
    return "ruby", 'require_relative "./config"\n', {"module": "config", "dots": "1"}


@case(tags=["import"])
def case_ruby_require_relative_parent() -> ImportCase:
    """require_relative "../lib/utils" — `../` becomes dots=2."""
    return "ruby", 'require_relative "../lib/utils"\n', {"module": "lib/utils", "dots": "2"}


@case(tags=["import"])
def case_ruby_require_empty_string() -> ImportCase:
    """require "" — empty string returns empty metadata.

    Covers `ruby.py` lines 53 and 70.
    """
    return "ruby", 'require ""\n', {}


# ── Multi-import ─────────────────────────────────────────────────────


@case(tags=["multi_import"])
def case_ruby_multiple_requires() -> MultiImportCase:
    """require + require_relative."""
    src = """\
require "json"
require_relative "helpers"
"""
    return "ruby", src, 2, [{"module": "json"}, {"module": "helpers", "dots": "1"}]


# ── Mixed ────────────────────────────────────────────────────────────


@case(tags=["mixed"])
def case_ruby_full_file() -> MixedCase:
    """Realistic Ruby file with doc comments on top-level
    declarations.  Comments inside the class body are not
    attached to their methods by the current Ruby grammar (see
    note in `case_docstrings.py`), so only the top-level
    module, class, and `main` carry docs here.  Methods carry the
    full module::class path now that addressing composes the
    enclosing-scope chain.
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
        [("start", "App::Server"), ("stop", "App::Server"), ("default", "App::Server")],
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
def case_ruby_constant() -> SymbolCase:
    """Top-level constant."""
    return "ruby", "MAX_SIZE = 100\n", [("variable", "MAX_SIZE", "")]


@case(tags=["symbol"])
def case_bash_assignment() -> SymbolCase:
    """Top-level variable assignment."""
    return "bash", "MAX=100\n", [("variable", "MAX", "")]


# ═════════════════════════════════════════════════════════════════════
# Module-level destructuring & multiple assignment (flat)
# ═════════════════════════════════════════════════════════════════════

_xfail_nested = pytest.mark.xfail(
    reason="nested/chained destructuring unsupported — no query-only recursion",
    strict=True,
)


@case(tags=["symbol"])
def case_ruby_multiple_assignment() -> SymbolCase:
    """Ruby multiple assignment of constants."""
    return "ruby", "A, B = 1, 2\n", [("variable", "A", ""), ("variable", "B", "")]


@case(tags=["symbol"])
def case_ruby_splat_assignment() -> SymbolCase:
    """Ruby splat target."""
    return "ruby", "A, *B = list\n", [("variable", "A", ""), ("variable", "B", "")]


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
def case_ruby_nested_unpack_xfail() -> SymbolCase:
    """Ruby nested destructuring — only the outer level captured today."""
    return (
        "ruby",
        "(A, B), C = x\n",
        [("variable", "A", ""), ("variable", "B", ""), ("variable", "C", "")],
    )
