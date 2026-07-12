"""C++ extraction test cases."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]


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


@case(tags=["symbol"])
def case_cpp_global() -> SymbolCase:
    """File-scope global with initialiser."""
    return "cpp", "int g = 5;\n", [("variable", "g", "")]
