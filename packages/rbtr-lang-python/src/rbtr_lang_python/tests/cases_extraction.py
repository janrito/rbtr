"""Python extraction test cases."""

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


@case(tags=["multi_import"])
def case_py_multiple_imports() -> MultiImportCase:
    """Two bare import statements."""
    src = """\
import os
import sys
"""
    return "python", src, 2, [{"module": "os"}, {"module": "sys"}]


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
