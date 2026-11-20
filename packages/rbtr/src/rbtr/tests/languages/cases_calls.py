"""Source files that call and extend, and what each call and base reaches.

Each case is files in the test module's `code` language (Python's grammar
and query under `.code`) or `pkg` (the same, its directories' files
sharing their names), and, for each target (a file and the symbol
`find-refs` is asked for), the calls and bases that reach it: the
referrer's file, its name and the edge. A strict `xfail` case states what
names-only resolution gets wrong (ARCHITECTURE, "Calls and bases").
A rule a plugin's own tests hold through a real language (member order,
members in scope, linked functions, static imports, re-exports) is not
repeated here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest_cases import case

from rbtr.testing.assertions import Referrer, Target


@dataclass(frozen=True)
class CallScenario:
    """Files, and for each target `(path, symbol)` the referrers `(path, name, edge)` that reach it."""

    files: dict[str, str]
    reached: dict[Target, set[Referrer]]


def case_a_call_reaches_a_definition_in_its_own_scope() -> CallScenario:
    """A bare name reaches the definition in the innermost enclosing scope; a method is never in scope."""
    return CallScenario(
        files={
            "a.code": """\
def run():
    inner()


def outer():
    def inner():
        pass

    inner()


def inner():
    pass


class C:
    def helper(self):
        pass


helper()
run()
"""
        },
        reached={
            ("a.code", "outer.inner"): {("a.code", "outer", "calls")},
            ("a.code", "inner"): {("a.code", "run", "calls"), ("a.code", "outer", "calls")},
            ("a.code", "run"): {("a.code", "", "calls")},
            ("a.code", "C.helper"): set(),
        },
    )


def case_a_call_reaches_what_its_files_imports_bind() -> CallScenario:
    """An imported name, an alias and a module's member each reach the definition they bind."""
    return CallScenario(
        files={
            "a.code": """\
from lib import f
from lib import h as g
import lib


def run():
    f()
    g()
    lib.h()
""",
            "lib.code": """\
def f():
    pass


def h():
    pass
""",
        },
        reached={
            ("lib.code", "f"): {("a.code", "run", "calls")},
            ("lib.code", "h"): {("a.code", "run", "calls")},
        },
    )


def case_two_files_bind_one_name_to_different_modules() -> CallScenario:
    """Each file's binding is its own: `g` in `a` is `one`'s `f`, `g` in `b` is `two`'s `f`."""
    return CallScenario(
        files={
            "a.code": """\
from one import f as g


def run():
    g()
""",
            "b.code": """\
from two import f as g


def run():
    g()
""",
            "one.code": """\
def f():
    return 1
""",
            "two.code": """\
def f():
    return 2
""",
        },
        reached={
            ("one.code", "f"): {("a.code", "run", "calls")},
            ("two.code", "f"): {("b.code", "run", "calls")},
        },
    )


def case_an_unbound_name_is_looked_up_in_star_imports_in_order() -> CallScenario:
    """A name nothing else binds is the first star-imported module's that has it."""
    return CallScenario(
        files={
            "a.code": """\
from one import *
from two import *


def run():
    f()
    g()
""",
            "one.code": """\
def f():
    return 1
""",
            "two.code": """\
def f():
    return 2


def g():
    pass
""",
        },
        reached={
            ("one.code", "f"): {("a.code", "run", "calls")},
            ("two.code", "f"): set(),
            ("two.code", "g"): {("a.code", "run", "calls")},
        },
    )


def case_self_and_super_reach_members_through_bases() -> CallScenario:
    """`self.m` is the class's own member, else its nearest base's; `super().m` starts at the bases."""
    return CallScenario(
        files={
            "a.code": """\
from base import Base


class Order(Base):
    def save(self):
        super().save()

    def run(self):
        self.save()
        self.check()
""",
            "base.code": """\
class Base:
    def save(self):
        pass

    def check(self):
        pass
""",
        },
        reached={
            ("base.code", "Base"): {("a.code", "Order", "inherits")},
            ("base.code", "Base.save"): {("a.code", "save", "calls")},
            ("base.code", "Base.check"): {("a.code", "run", "calls")},
            ("a.code", "Order.save"): {("a.code", "run", "calls")},
        },
    )


def case_a_call_on_a_typed_value_reaches_its_classs_method() -> CallScenario:
    """`order = Order()` then `order.save()`: the value is an `Order`, so the call is `Order.save`."""
    return CallScenario(
        files={
            "a.code": """\
class Order:
    def save(self):
        pass


def run():
    order = Order()
    order.save()
"""
        },
        reached={
            ("a.code", "Order"): {("a.code", "run", "calls")},
            ("a.code", "Order.save"): {("a.code", "run", "calls")},
        },
    )


def case_calling_a_typed_value_itself_reaches_nothing() -> CallScenario:
    """`order()` with `order: Order` calls the instance, not the class: no edge."""
    return CallScenario(
        files={
            "a.code": """\
class Order:
    pass


def run(order: Order):
    order()
""",
        },
        reached={("a.code", "Order"): set()},
    )


def case_a_call_on_a_value_reaches_nothing() -> CallScenario:
    """`def run(f): return f()` beside a module-level `f`: the call is on the parameter, which makes no edge."""
    return CallScenario(
        files={
            "a.code": """\
def f():
    pass


def run(f):
    return f()
""",
        },
        reached={("a.code", "f"): set()},
    )


def case_a_name_an_import_lists_may_be_a_submodule() -> CallScenario:
    """`from pkg import mod` binds the submodule when `pkg` defines no `mod`; `mod.f` is its member."""
    return CallScenario(
        files={
            "a.code": """\
from pkg import mod


def run():
    mod.f()
""",
            "pkg/__init__.code": "",
            "pkg/mod.code": """\
def f():
    pass
""",
        },
        reached={("pkg/mod.code", "f"): {("a.code", "run", "calls")}},
    )


def case_a_chunk_at_two_paths_calls_from_each() -> CallScenario:
    """Byte-identical files share their chunks; each copy's call reaches its own sibling."""
    caller = """\
from .b import f


def run():
    f()
"""
    return CallScenario(
        files={
            "lib/a.code": caller,
            "lib/b.code": """\
def f():
    return 1
""",
            "src/a.code": caller,
            "src/b.code": """\
def f():
    return 2
""",
        },
        reached={
            ("lib/b.code", "f"): {("lib/a.code", "run", "calls")},
            ("src/b.code", "f"): {("src/a.code", "run", "calls")},
        },
    )


def case_files_in_one_directory_share_their_names_where_directories_scope() -> CallScenario:
    """In `pkg`, `helper()` is the `helper` another file of its directory defines, not another directory's."""
    return CallScenario(
        files={
            "pkg/a.pkg": """\
def run():
    helper()
""",
            "pkg/b.pkg": """\
def helper():
    return 1
""",
            "other/c.pkg": """\
def helper():
    return 2
""",
        },
        reached={
            ("pkg/b.pkg", "helper"): {("pkg/a.pkg", "run", "calls")},
            ("other/c.pkg", "helper"): set(),
        },
    )


@case(
    marks=pytest.mark.xfail(
        strict=True,
        reason="names only: a local's type is what is written beside it, so one assigned "
        "from a factory has the factory's name as its type",
    )
)
def case_a_local_assigned_from_a_factory_reaches_the_class_it_returns() -> CallScenario:
    """`order = make_order()` then `order.save()`: `make_order` returns an `Order`, so the call is `Order.save`."""
    return CallScenario(
        files={
            "a.code": """\
class Order:
    def save(self):
        pass


def make_order():
    return Order()


def run():
    order = make_order()
    order.save()
"""
        },
        reached={("a.code", "Order.save"): {("a.code", "run", "calls")}},
    )
