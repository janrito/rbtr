"""C extraction test cases."""

from __future__ import annotations

from pytest_cases import case

type SymbolCase = tuple[str, str, list[tuple[str, str, str]]]
type ImportCase = tuple[str, str, dict[str, str | list[dict[str, str]]]]
type MultiImportCase = tuple[str, str, int, list[dict[str, str]]]
type MixedCase = tuple[str, str, set[str], list[tuple[str, str]]]


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


@case(tags=["import"])
def case_c_include_system() -> ImportCase:
    """#include <stdio.h>."""
    return "c", "#include <stdio.h>\n", {"module": "stdio.h", "names": "*"}


@case(tags=["import"])
def case_c_include_local() -> ImportCase:
    """#include "mylib.h"."""
    return "c", '#include "mylib.h"\n', {"module": "mylib.h", "names": "*"}


@case(tags=["import"])
def case_c_include_nested_path() -> ImportCase:
    """#include "utils/helpers.h"."""
    return "c", '#include "utils/helpers.h"\n', {"module": "utils/helpers.h", "names": "*"}


@case(tags=["import"])
def case_c_include_system_nested() -> ImportCase:
    """#include <sys/types.h>."""
    return "c", "#include <sys/types.h>\n", {"module": "sys/types.h", "names": "*"}


@case(tags=["multi_import"])
def case_c_multiple_includes() -> MultiImportCase:
    """Two include directives."""
    src = """\
#include <stdlib.h>
#include "local.h"
"""
    return "c", src, 2, [{"module": "stdlib.h", "names": "*"}, {"module": "local.h", "names": "*"}]


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


@case(tags=["symbol"])
def case_c_global() -> SymbolCase:
    """File-scope global with initialiser."""
    return "c", "int g = 5;\n", [("variable", "g", "")]


type ReferencesCase = tuple[str, str, list[tuple[str, str, list[str], list[str]]]]


@case(tags=["references"])
def case_c_calls() -> ReferencesCase:
    """A call by name, and calls through a struct's field, which start at a value."""
    src = """\
int run(struct shop *s) {
    helper(1);
    s->check();
    s->items.size();
    return 0;
}
"""
    return (
        "c",
        src,
        [
            (
                "function",
                "run",
                ["unqualified:helper", "unqualified:s.check", "unqualified:s.items.size"],
                [],
            )
        ],
    )


@case(tags=["symbol"])
def case_c_functions_returning_pointers() -> SymbolCase:
    """A return type's `*` or `**` wraps the declarator; each function and prototype is still extracted."""
    src = """\
char* top(void) { return 0; }
char **names(void) { return 0; }
const char *prototype(int n);
"""
    return (
        "c",
        src,
        [
            ("function", "top", ""),
            ("function", "names", ""),
            ("function", "prototype", ""),
        ],
    )


@case(tags=["symbol"])
def case_c_a_function_returning_a_function_pointer() -> SymbolCase:
    """`void (*handler(int sig))(int)`: named inside the pointer, as `signal` is declared."""
    src = "void (*handler(int sig))(int) { return reset; }\n"
    return "c", src, [("function", "handler", "")]


@case(tags=["symbol"])
def case_c_a_function_named_by_a_call_shaped_declarator() -> SymbolCase:
    """`double PREFIX(predict)(…)` and a C++ header's `operator()(…)` read as C are named as written."""
    src = """\
double PREFIX(predict)(int x) { return 0; }
bool operator()(int x) { return x; }
"""
    return "c", src, [("function", "PREFIX(predict)", ""), ("function", "operator()", "")]


@case(tags=["symbol"])
def case_c_declarations_inside_an_include_guard() -> SymbolCase:
    """A header's prototypes and variables inside `#ifndef` / `#if` / `#else` are its own."""
    src = """\
#ifndef GREETER_H
#define GREETER_H
char *greet(const char *name);
extern int greeted;
#if WITH_LOUD
void shout(void);
#else
void whisper(void);
#endif
#endif
"""
    return (
        "c",
        src,
        [
            ("function", "greet", ""),
            ("variable", "greeted", ""),
            ("function", "shout", ""),
            ("function", "whisper", ""),
        ],
    )


@case(tags=["symbol"])
def case_c_a_local_declaration_is_no_definition() -> SymbolCase:
    """A declaration inside a function body is the function's, not a chunk of its own."""
    src = "int run(void) {\n  int count = 0;\n  return count;\n}\n"
    return "c", src, [("function", "run", "")]
