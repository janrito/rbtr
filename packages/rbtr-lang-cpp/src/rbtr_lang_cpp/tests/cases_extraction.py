"""C++ extraction test cases."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str | list[dict[str, str]]]]
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
    return "cpp", "#include <iostream>\n", {"module": "iostream", "names": "*"}


@case(tags=["import"])
def case_cpp_include_local() -> ImportCase:
    """#include "myheader.h"."""
    return "cpp", '#include "myheader.h"\n', {"module": "myheader.h", "names": "*"}


@case(tags=["import"])
def case_cpp_include_nested() -> ImportCase:
    """#include <boost/optional.hpp>."""
    return "cpp", "#include <boost/optional.hpp>\n", {"module": "boost/optional.hpp", "names": "*"}


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


type ReferencesCase = tuple[str, str, list[tuple[str, str, list[str], list[str]]]]


@case(tags=["references"])
def case_cpp_calls_and_bases() -> ReferencesCase:
    """Each call shape and the bases after their access specifiers; an out-of-class member is scoped by its class."""
    src = """\
namespace shop {
class Order : public Base, private mixins::Logged {
    int check() { return 1; }
};
void Order::save() {
    Base::save();
    this->check();
    check();
    util::log();
    make<int>();
    new Foo();
}
}
"""
    return (
        "cpp",
        src,
        [
            ("class", "Order", [], ["unqualified:Base", "unqualified:mixins.Logged"]),
            (
                "function",
                "save",
                [
                    "unqualified:Base.save",
                    "member:check",
                    "unqualified:check",
                    "unqualified:util.log",
                    "unqualified:make",
                    "unqualified:Foo",
                ],
                [],
            ),
        ],
    )


@case(tags=["symbol"])
def case_cpp_functions_returning_pointers_and_references() -> SymbolCase:
    """A return type's `*`, `**`, `&` or `&&` wraps the declarator; each function is still extracted."""
    src = """\
int* top() { return 0; }
char** names() { return 0; }
int& ref() { static int x; return x; }
int&& moved();
char* prototype();
namespace shop {
int* find();
}
class K {
 public:
  char* m() { return 0; }
  int& n();
  char* declared();
  K& operator=(const K&) { return *this; }
};
char* K::declared() { return 0; }
int& K::n() { static int x; return x; }
"""
    return (
        "cpp",
        src,
        [
            ("function", "top", ""),
            ("function", "names", ""),
            ("function", "ref", ""),
            ("function", "moved", ""),
            ("function", "prototype", ""),
            ("function", "find", "shop"),
            ("method", "m", "K"),
            ("method", "n", "K"),
            ("method", "declared", "K"),
            ("method", "operator=", "K"),
        ],
    )


@case(tags=["symbol"])
def case_cpp_definitions_named_other_than_by_an_identifier() -> SymbolCase:
    """Destructors, operators, template specialisations and functions returning function pointers."""
    src = """\
class Order {
 public:
  Order();
  ~Order();
  bool operator==(const Order& o) const;
};
Order::~Order() { close(); }
Order& Order::operator<<(int x) { return *this; }
bool Order::operator==(const Order& o) const { return true; }
template <> void encode<int>(int x) { write(x); }
void (*handler(int sig))(int) { return reset; }
class Box {
 public:
  ~Box() { drop(); }
};
namespace shop {
class Cart {
  ~Cart();
};
}
shop::Cart::~Cart() {}
"""
    return (
        "cpp",
        src,
        [
            ("method", "Order", "Order"),
            ("method", "~Order", "Order"),
            ("method", "operator==", "Order"),
            ("function", "~Order", "Order"),
            ("function", "operator<<", "Order"),
            ("function", "operator==", "Order"),
            ("function", "encode", ""),
            ("function", "handler", ""),
            ("method", "~Box", "Box"),
            ("function", "~Cart", "shop::Cart"),
        ],
    )


@case(tags=["symbol"])
def case_cpp_a_function_named_by_a_macro() -> SymbolCase:
    """`double PREFIX(predict)(…)`: the preprocessor builds the name, so it is named as written."""
    src = """\
double PREFIX(predict)(const model* m) { return 0; }
const char *PREFIX(check)(int x) { return 0; }
"""
    return (
        "cpp",
        src,
        [("function", "PREFIX(predict)", ""), ("function", "PREFIX(check)", "")],
    )


@case(tags=["symbol"])
def case_cpp_declarations_inside_guards_and_extern_c() -> SymbolCase:
    """Prototypes and variables inside `#ifdef` and `extern "C"` are the header's own."""
    src = """\
#ifndef API_H
#define API_H
#ifdef __cplusplus
extern "C" {
#endif
int api_open(const char *path);
extern int api_errno;
#ifdef __cplusplus
}
#endif
namespace shop {
int count;
}
#endif
"""
    return (
        "cpp",
        src,
        [
            ("function", "api_open", ""),
            ("variable", "api_errno", ""),
            ("variable", "count", "shop"),
        ],
    )
