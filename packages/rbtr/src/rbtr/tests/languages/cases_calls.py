"""Chunk rows recording calls and bases, and the `calls` and `inherits` edges they make.

Each case writes every `EdgeChunkRow` column for the neutral languages of
the test module's `resolutions` fixture: `code` (path modules, `.code`),
`dotted` (dotted modules, `.dot`), `members` (implicit `this`),
`package` (a directory's files share their names) and `linked` (a
function is one across files by its scope and name). No plugin is involved, so each case
states one rule of how a call or base reaches a definition, whatever
language wrote it. A strict `xfail` case states what names-only resolution
gets wrong (ARCHITECTURE, "Calls and bases").
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest_cases import case


@dataclass(frozen=True)
class CallScenario:
    """`EdgeChunkRow` columns, the repository's files, and the call and inherits edges inferred.

    An edge is `(source id, target id, kind)`; import edges are left out.
    `located`, when given, is each call or inherits edge's `(source path,
    target path)`, for chunks a case places at several paths.
    """

    rows: dict[str, list]
    repo_files: frozenset[str]
    expected: frozenset[tuple[str, str, str]]
    located: frozenset[tuple[str, str]] | None = None


def case_a_call_reaches_a_definition_in_its_own_scope() -> CallScenario:
    """A bare name reaches the definition in the innermost enclosing scope; a method is never in scope."""
    return CallScenario(
        rows={
            "id": ["run", "outer", "nested_inner", "top_inner", "C", "C.helper", "frag"],
            "file_path": ["a.code"] * 7,
            "line_start": [1, 2, 3, 9, 11, 12, 20],
            "kind": ["function", "function", "function", "function", "class", "method", "fragment"],
            "name": ["run", "outer", "inner", "inner", "C", "helper", ""],
            "scope": ["", "", "outer", "", "", "C", ""],
            "language": ["code"] * 7,
            "module": [None] * 7,
            "names": [None] * 7,
            "language_hint": [None] * 7,
            "bindings": [None] * 7,
            "refs": [
                [{"kind": "calls", "lookup": "unqualified", "path": ["inner"]}],
                [{"kind": "calls", "lookup": "unqualified", "path": ["inner"]}],
                [],
                [],
                [],
                [],
                [
                    {"kind": "calls", "lookup": "unqualified", "path": ["helper"]},
                    {"kind": "calls", "lookup": "unqualified", "path": ["run"]},
                ],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset(
            {
                ("run", "top_inner", "calls"),
                ("outer", "nested_inner", "calls"),
                ("frag", "run", "calls"),
            }
        ),
    )


def case_a_call_reaches_what_its_files_imports_bind() -> CallScenario:
    """An imported name, an alias and a module's member each reach the definition they bind."""
    return CallScenario(
        rows={
            "id": ["imp_f", "imp_g", "imp_lib", "run", "f", "h"],
            "file_path": ["a.code", "a.code", "a.code", "a.code", "lib.code", "lib.code"],
            "line_start": [1, 2, 3, 5, 1, 3],
            "kind": ["import", "import", "import", "function", "function", "function"],
            "name": ["from lib import f", "from lib import h as g", "import lib", "run", "f", "h"],
            "scope": [""] * 6,
            "language": ["code"] * 6,
            "module": ["lib.code", "lib.code", "lib.code", None, None, None],
            "names": ["f", "h", "", None, None, None],
            "language_hint": ["", "", "", None, None, None],
            "bindings": [
                [{"name": "f", "local": "f"}],
                [{"name": "h", "local": "g"}],
                [{"name": "lib.code", "local": "lib"}],
                None,
                None,
                None,
            ],
            "refs": [
                None,
                None,
                None,
                [
                    {"kind": "calls", "lookup": "unqualified", "path": ["f"]},
                    {"kind": "calls", "lookup": "unqualified", "path": ["g"]},
                    {"kind": "calls", "lookup": "unqualified", "path": ["lib", "h"]},
                ],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code", "lib.code"}),
        expected=frozenset({("run", "f", "calls"), ("run", "h", "calls")}),
    )


def case_two_files_bind_one_name_to_different_modules() -> CallScenario:
    """Each file's binding is its own: `g` in `a` is `one`'s `f`, `g` in `b` is `two`'s `f`."""
    return CallScenario(
        rows={
            "id": ["a_imp", "a_run", "b_imp", "b_run", "one_f", "two_f"],
            "file_path": ["a.code", "a.code", "b.code", "b.code", "one.code", "two.code"],
            "line_start": [1, 3, 1, 3, 1, 1],
            "kind": ["import", "function", "import", "function", "function", "function"],
            "name": ["from one import f as g", "run", "from two import f as g", "run", "f", "f"],
            "scope": [""] * 6,
            "language": ["code"] * 6,
            "module": ["one.code", None, "two.code", None, None, None],
            "names": ["f", None, "f", None, None, None],
            "language_hint": ["", None, "", None, None, None],
            "bindings": [
                [{"name": "f", "local": "g"}],
                None,
                [{"name": "f", "local": "g"}],
                None,
                None,
                None,
            ],
            "refs": [
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["g"]}],
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["g"]}],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code", "b.code", "one.code", "two.code"}),
        expected=frozenset({("a_run", "one_f", "calls"), ("b_run", "two_f", "calls")}),
    )


def case_alternative_imports_of_one_name_reach_each() -> CallScenario:
    """`try: import ujson as json` / `except: import json` binds `json` twice; either may run, so both are reached."""
    return CallScenario(
        rows={
            "id": ["imp_ujson", "imp_json", "run", "ujson_loads", "json_loads"],
            "file_path": ["a.code", "a.code", "a.code", "ujson.code", "json.code"],
            "line_start": [2, 4, 7, 1, 1],
            "kind": ["import", "import", "function", "function", "function"],
            "name": ["import ujson as json", "import json", "run", "loads", "loads"],
            "scope": [""] * 5,
            "language": ["code"] * 5,
            "module": ["ujson.code", "json.code", None, None, None],
            "names": ["", "", None, None, None],
            "language_hint": ["", "", None, None, None],
            "bindings": [
                [{"name": "ujson.code", "local": "json"}],
                [{"name": "json.code", "local": "json"}],
                None,
                None,
                None,
            ],
            "refs": [
                None,
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["json", "loads"]}],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code", "ujson.code", "json.code"}),
        expected=frozenset({("run", "ujson_loads", "calls"), ("run", "json_loads", "calls")}),
    )


@case(
    marks=pytest.mark.xfail(
        strict=True,
        reason="names only: a later import binding the same name shadows the earlier, "
        "but resolution keeps no order and reaches both",
    )
)
def case_a_later_import_of_a_name_shadows_an_earlier() -> CallScenario:
    """`from one import f as g` then `from two import h as g`: `g` is `two`'s `h` alone."""
    return CallScenario(
        rows={
            "id": ["imp_one", "imp_two", "run", "one_f", "two_h"],
            "file_path": ["a.code", "a.code", "a.code", "one.code", "two.code"],
            "line_start": [1, 2, 4, 1, 1],
            "kind": ["import", "import", "function", "function", "function"],
            "name": ["from one import f as g", "from two import h as g", "run", "f", "h"],
            "scope": [""] * 5,
            "language": ["code"] * 5,
            "module": ["one.code", "two.code", None, None, None],
            "names": ["f", "h", None, None, None],
            "language_hint": ["", "", None, None, None],
            "bindings": [
                [{"name": "f", "local": "g"}],
                [{"name": "h", "local": "g"}],
                None,
                None,
                None,
            ],
            "refs": [
                None,
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["g"]}],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code", "one.code", "two.code"}),
        expected=frozenset({("run", "two_h", "calls")}),
    )


def case_a_reexported_name_is_followed_to_its_definition() -> CallScenario:
    """A module that imports a name rather than defining it passes the lookup on; a cycle ends it."""
    return CallScenario(
        rows={
            "id": ["imp_f", "imp_g", "run", "mid_f", "f", "x_g", "y_g"],
            "file_path": ["a.code", "a.code", "a.code", "mid.code", "lib.code", "x.code", "y.code"],
            "line_start": [1, 2, 4, 1, 1, 1, 1],
            "kind": ["import", "import", "function", "import", "function", "import", "import"],
            "name": [
                "from mid import f",
                "from x import g",
                "run",
                "from lib import f",
                "f",
                "from y import g",
                "from x import g",
            ],
            "scope": [""] * 7,
            "language": ["code"] * 7,
            "module": ["mid.code", "x.code", None, "lib.code", None, "y.code", "x.code"],
            "names": ["f", "g", None, "f", None, "g", "g"],
            "language_hint": ["", "", None, "", None, "", ""],
            "bindings": [
                [{"name": "f", "local": "f"}],
                [{"name": "g", "local": "g"}],
                None,
                [{"name": "f", "local": "f"}],
                None,
                [{"name": "g", "local": "g"}],
                [{"name": "g", "local": "g"}],
            ],
            "refs": [
                None,
                None,
                [
                    {"kind": "calls", "lookup": "unqualified", "path": ["f"]},
                    {"kind": "calls", "lookup": "unqualified", "path": ["g"]},
                ],
                None,
                [],
                None,
                None,
            ],
        },
        repo_files=frozenset({"a.code", "mid.code", "lib.code", "x.code", "y.code"}),
        expected=frozenset({("run", "f", "calls")}),
    )


def case_an_unbound_name_is_looked_up_in_star_imports_in_order() -> CallScenario:
    """A name nothing else binds is the first star-imported module's that has it."""
    return CallScenario(
        rows={
            "id": ["star_one", "star_two", "run", "one_f", "two_f", "two_g"],
            "file_path": ["a.code", "a.code", "a.code", "one.code", "two.code", "two.code"],
            "line_start": [1, 2, 4, 1, 1, 3],
            "kind": ["import", "import", "function", "function", "function", "function"],
            "name": ["from one import *", "from two import *", "run", "f", "f", "g"],
            "scope": [""] * 6,
            "language": ["code"] * 6,
            "module": ["one.code", "two.code", None, None, None, None],
            "names": ["*", "*", None, None, None, None],
            "language_hint": ["", "", None, None, None, None],
            "bindings": [[], [], None, None, None, None],
            "refs": [
                None,
                None,
                [
                    {"kind": "calls", "lookup": "unqualified", "path": ["f"]},
                    {"kind": "calls", "lookup": "unqualified", "path": ["g"]},
                ],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code", "one.code", "two.code"}),
        expected=frozenset({("run", "one_f", "calls"), ("run", "two_g", "calls")}),
    )


def case_self_and_super_reach_members_through_bases() -> CallScenario:
    """`self.m` is the class's own member, else its nearest base's; `super().m` starts at the bases."""
    return CallScenario(
        rows={
            "id": [
                "imp_base",
                "Order",
                "Order.save",
                "Order.run",
                "Base",
                "Base.save",
                "Base.check",
            ],
            "file_path": [
                "a.code",
                "a.code",
                "a.code",
                "a.code",
                "base.code",
                "base.code",
                "base.code",
            ],
            "line_start": [1, 3, 4, 7, 1, 2, 5],
            "kind": ["import", "class", "method", "method", "class", "method", "method"],
            "name": ["from base import Base", "Order", "save", "run", "Base", "save", "check"],
            "scope": ["", "", "Order", "Order", "", "Base", "Base"],
            "language": ["code"] * 7,
            "module": ["base.code", None, None, None, None, None, None],
            "names": ["Base", None, None, None, None, None, None],
            "language_hint": ["", None, None, None, None, None, None],
            "bindings": [[{"name": "Base", "local": "Base"}], None, None, None, None, None, None],
            "refs": [
                None,
                [{"kind": "inherits", "lookup": "unqualified", "path": ["Base"]}],
                [{"kind": "calls", "lookup": "base", "path": ["save"]}],
                [
                    {"kind": "calls", "lookup": "member", "path": ["save"]},
                    {"kind": "calls", "lookup": "member", "path": ["check"]},
                ],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code", "base.code"}),
        expected=frozenset(
            {
                ("Order", "Base", "inherits"),
                ("Order.save", "Base.save", "calls"),
                ("Order.run", "Order.save", "calls"),
                ("Order.run", "Base.check", "calls"),
            }
        ),
    )


def case_the_first_listed_of_two_bases_defining_a_member_wins() -> CallScenario:
    """`class C(A, B)`, both defining `m`: `self.m` in `C` is `A.m`."""
    return CallScenario(
        rows={
            "id": ["C", "C.run", "A", "A.m", "B", "B.m"],
            "file_path": ["a.code"] * 6,
            "line_start": [1, 2, 5, 6, 9, 10],
            "kind": ["class", "method", "class", "method", "class", "method"],
            "name": ["C", "run", "A", "m", "B", "m"],
            "scope": ["", "C", "", "A", "", "B"],
            "language": ["code"] * 6,
            "module": [None] * 6,
            "names": [None] * 6,
            "language_hint": [None] * 6,
            "bindings": [None] * 6,
            "refs": [
                [
                    {"kind": "inherits", "lookup": "unqualified", "path": ["A"]},
                    {"kind": "inherits", "lookup": "unqualified", "path": ["B"]},
                ],
                [{"kind": "calls", "lookup": "member", "path": ["m"]}],
                [],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset(
            {("C", "A", "inherits"), ("C", "B", "inherits"), ("C.run", "A.m", "calls")}
        ),
    )


def case_a_bases_base_comes_before_a_later_base() -> CallScenario:
    """`class C(A, B)`, `class A(X)`, `X` and `B` defining `m`: the order is C, A, X, B, so `self.m` is `X.m`."""
    return CallScenario(
        rows={
            "id": ["C", "C.run", "A", "X", "X.m", "B", "B.m"],
            "file_path": ["a.code"] * 7,
            "line_start": [1, 2, 5, 7, 8, 11, 12],
            "kind": ["class", "method", "class", "class", "method", "class", "method"],
            "name": ["C", "run", "A", "X", "m", "B", "m"],
            "scope": ["", "C", "", "", "X", "", "B"],
            "language": ["code"] * 7,
            "module": [None] * 7,
            "names": [None] * 7,
            "language_hint": [None] * 7,
            "bindings": [None] * 7,
            "refs": [
                [
                    {"kind": "inherits", "lookup": "unqualified", "path": ["A"]},
                    {"kind": "inherits", "lookup": "unqualified", "path": ["B"]},
                ],
                [{"kind": "calls", "lookup": "member", "path": ["m"]}],
                [{"kind": "inherits", "lookup": "unqualified", "path": ["X"]}],
                [],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset(
            {
                ("C", "A", "inherits"),
                ("C", "B", "inherits"),
                ("A", "X", "inherits"),
                ("C.run", "X.m", "calls"),
            }
        ),
    )


def case_a_call_on_a_typed_value_reaches_its_classs_method() -> CallScenario:
    """`order = Order()` then `order.save()`: the value is an `Order`, so the call is `Order.save`."""
    return CallScenario(
        rows={
            "id": ["Order", "Order.save", "run"],
            "file_path": ["a.code"] * 3,
            "line_start": [1, 2, 5],
            "kind": ["class", "method", "function"],
            "name": ["Order", "save", "run"],
            "scope": ["", "Order", ""],
            "language": ["code"] * 3,
            "module": [None] * 3,
            "names": [None] * 3,
            "language_hint": [None] * 3,
            "bindings": [None] * 3,
            "refs": [
                [],
                [],
                [
                    {"kind": "calls", "lookup": "unqualified", "path": ["Order"], "type": None},
                    {
                        "kind": "calls",
                        "lookup": "type",
                        "path": ["order", "save"],
                        "type": ["Order"],
                    },
                ],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset({("run", "Order", "calls"), ("run", "Order.save", "calls")}),
    )


def case_calling_a_typed_value_itself_reaches_nothing() -> CallScenario:
    """`order()` with `order: Order` calls the instance, not the class: no edge."""
    return CallScenario(
        rows={
            "id": ["Order", "run"],
            "file_path": ["a.code"] * 2,
            "line_start": [1, 4],
            "kind": ["class", "function"],
            "name": ["Order", "run"],
            "scope": ["", ""],
            "language": ["code"] * 2,
            "module": [None] * 2,
            "names": [None] * 2,
            "language_hint": [None] * 2,
            "bindings": [None] * 2,
            "refs": [
                [],
                [{"kind": "calls", "lookup": "type", "path": ["order"], "type": ["Order"]}],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset(),
    )


def case_a_call_on_a_value_reaches_nothing() -> CallScenario:
    """`def run(f): return f()` beside a module-level `f`: extraction marks the call as on a value, which makes no edge."""
    return CallScenario(
        rows={
            "id": ["f", "run"],
            "file_path": ["a.code"] * 2,
            "line_start": [1, 4],
            "kind": ["function", "function"],
            "name": ["f", "run"],
            "scope": ["", ""],
            "language": ["code"] * 2,
            "module": [None] * 2,
            "names": [None] * 2,
            "language_hint": [None] * 2,
            "bindings": [None] * 2,
            "refs": [
                [],
                [{"kind": "calls", "lookup": "type", "path": ["f"], "type": None}],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset(),
    )


def case_a_name_an_import_lists_may_be_a_submodule() -> CallScenario:
    """`from pkg import mod` binds the submodule when `pkg` defines no `mod`; `mod.f` is its member."""
    return CallScenario(
        rows={
            "id": ["imp_mod", "run", "f"],
            "file_path": ["a.dot", "a.dot", "pkg/mod.dot"],
            "line_start": [1, 3, 1],
            "kind": ["import", "function", "function"],
            "name": ["from pkg import mod", "run", "f"],
            "scope": [""] * 3,
            "language": ["dotted"] * 3,
            "module": ["pkg", None, None],
            "names": ["mod", None, None],
            "language_hint": ["", None, None],
            "bindings": [[{"name": "mod", "local": "mod"}], None, None],
            "refs": [None, [{"kind": "calls", "lookup": "unqualified", "path": ["mod", "f"]}], []],
        },
        repo_files=frozenset({"a.dot", "pkg/init.dot", "pkg/mod.dot"}),
        expected=frozenset({("run", "f", "calls")}),
    )


def case_a_chunk_at_two_paths_calls_from_each() -> CallScenario:
    """Byte-identical files share their chunks; each copy's call reaches its own sibling."""
    return CallScenario(
        rows={
            "id": ["imp", "run", "imp", "run", "f_src", "f_lib"],
            "file_path": [
                "src/a.code",
                "src/a.code",
                "lib/a.code",
                "lib/a.code",
                "src/b.code",
                "lib/b.code",
            ],
            "line_start": [1, 3, 1, 3, 1, 1],
            "kind": ["import", "function", "import", "function", "function", "function"],
            "name": ["from ./b import f", "run", "from ./b import f", "run", "f", "f"],
            "scope": [""] * 6,
            "language": ["code"] * 6,
            "module": ["./b.code", None, "./b.code", None, None, None],
            "names": ["f", None, "f", None, None, None],
            "language_hint": ["", None, "", None, None, None],
            "bindings": [
                [{"name": "f", "local": "f"}],
                None,
                [{"name": "f", "local": "f"}],
                None,
                None,
                None,
            ],
            "refs": [
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["f"]}],
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["f"]}],
                [],
                [],
            ],
        },
        repo_files=frozenset({"src/a.code", "lib/a.code", "src/b.code", "lib/b.code"}),
        expected=frozenset({("run", "f_src", "calls"), ("run", "f_lib", "calls")}),
        located=frozenset({("src/a.code", "src/b.code"), ("lib/a.code", "lib/b.code")}),
    )


def case_a_methods_unqualified_name_reaches_its_classs_member_where_members_are_in_scope() -> (
    CallScenario
):
    """In `members`, `helper()` in a method is its class's or a base's `helper`; in `code`, nothing."""
    return CallScenario(
        rows={
            "id": ["Base", "Base.helper", "C", "C.run", "K", "K.helper", "K.run"],
            "file_path": ["a.mem", "a.mem", "a.mem", "a.mem", "k.code", "k.code", "k.code"],
            "line_start": [1, 2, 5, 6, 1, 2, 5],
            "kind": ["class", "method", "class", "method", "class", "method", "method"],
            "name": ["Base", "helper", "C", "run", "K", "helper", "run"],
            "scope": ["", "Base", "", "C", "", "K", "K"],
            "language": ["members"] * 4 + ["code"] * 3,
            "module": [None] * 7,
            "names": [None] * 7,
            "language_hint": [None] * 7,
            "bindings": [None] * 7,
            "refs": [
                [],
                [],
                [{"kind": "inherits", "lookup": "unqualified", "path": ["Base"]}],
                [{"kind": "calls", "lookup": "unqualified", "path": ["helper"]}],
                [],
                [],
                [{"kind": "calls", "lookup": "unqualified", "path": ["helper"]}],
            ],
        },
        repo_files=frozenset({"a.mem", "k.code"}),
        expected=frozenset({("C", "Base", "inherits"), ("C.run", "Base.helper", "calls")}),
    )


def case_files_in_one_directory_share_their_names_where_directories_scope() -> CallScenario:
    """In `package`, `helper()` is the `helper` another file of its directory defines, not another directory's."""
    return CallScenario(
        rows={
            "id": ["run", "pkg_helper", "other_helper"],
            "file_path": ["pkg/a.pkg", "pkg/b.pkg", "other/c.pkg"],
            "line_start": [1, 1, 1],
            "kind": ["function", "function", "function"],
            "name": ["run", "helper", "helper"],
            "scope": ["", "", ""],
            "language": ["package"] * 3,
            "module": [None] * 3,
            "names": [None] * 3,
            "language_hint": [None] * 3,
            "bindings": [None] * 3,
            "refs": [[{"kind": "calls", "lookup": "unqualified", "path": ["helper"]}], [], []],
        },
        repo_files=frozenset({"pkg/a.pkg", "pkg/b.pkg", "other/c.pkg"}),
        expected=frozenset({("run", "pkg_helper", "calls")}),
    )


def case_a_function_reached_where_functions_are_linked_reaches_each_of_its_name() -> CallScenario:
    """In `linked`, `helper()` reaching the `helper` its include declares also reaches `impl.lnk`'s; not another scope's or language's."""
    return CallScenario(
        rows={
            "id": ["include", "run", "prototype", "definition", "namespaced", "unlinked"],
            "file_path": ["main.lnk", "main.lnk", "util.lnk", "impl.lnk", "ns.lnk", "other.code"],
            "line_start": [1, 3, 1, 1, 1, 1],
            "kind": ["import"] + ["function"] * 5,
            "name": ['#include "util.lnk"', "run", "helper", "helper", "helper", "helper"],
            "scope": ["", "", "", "", "ns", ""],
            "language": ["linked"] * 5 + ["code"],
            "module": ["util.lnk", None, None, None, None, None],
            "names": ["*", None, None, None, None, None],
            "language_hint": ["", None, None, None, None, None],
            "bindings": [[], None, None, None, None, None],
            "refs": [
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["helper"]}],
                [],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"main.lnk", "util.lnk", "impl.lnk", "ns.lnk", "other.code"}),
        expected=frozenset({("run", "prototype", "calls"), ("run", "definition", "calls")}),
    )


def case_a_base_listed_after_its_overrider_is_hidden_by_it() -> CallScenario:
    """`class D(B, A)`, `class B(A)`, `B.m` overriding `A.m`: the order is D, B, A, so `self.m` is `B.m`."""
    return CallScenario(
        rows={
            "id": ["D", "D.run", "B", "B.m", "A", "A.m"],
            "file_path": ["a.code"] * 6,
            "line_start": [1, 2, 4, 5, 7, 8],
            "kind": ["class", "method", "class", "method", "class", "method"],
            "name": ["D", "run", "B", "m", "A", "m"],
            "scope": ["", "D", "", "B", "", "A"],
            "language": ["code"] * 6,
            "module": [None] * 6,
            "names": [None] * 6,
            "language_hint": [None] * 6,
            "bindings": [None] * 6,
            "refs": [
                [
                    {"kind": "inherits", "lookup": "unqualified", "path": ["B"]},
                    {"kind": "inherits", "lookup": "unqualified", "path": ["A"]},
                ],
                [{"kind": "calls", "lookup": "member", "path": ["m"]}],
                [{"kind": "inherits", "lookup": "unqualified", "path": ["A"]}],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset(
            [
                ("B", "A", "inherits"),
                ("D", "A", "inherits"),
                ("D", "B", "inherits"),
                ("D.run", "B.m", "calls"),
            ]
        ),
    )


def case_a_reversed_language_searches_its_last_listed_base_first() -> CallScenario:
    """Ruby's order: `class C < Base` then `include Mixin`, both defining `m`: `self.m` is `Mixin.m`."""
    return CallScenario(
        rows={
            "id": ["C", "C.run", "Base", "Base.m", "Mixin", "Mixin.m"],
            "file_path": ["a.rev"] * 6,
            "line_start": [1, 2, 4, 5, 7, 8],
            "kind": ["class", "method", "class", "method", "class", "method"],
            "name": ["C", "run", "Base", "m", "Mixin", "m"],
            "scope": ["", "C", "", "Base", "", "Mixin"],
            "language": ["reversed"] * 6,
            "module": [None] * 6,
            "names": [None] * 6,
            "language_hint": [None] * 6,
            "bindings": [None] * 6,
            "refs": [
                [
                    {"kind": "inherits", "lookup": "unqualified", "path": ["Base"]},
                    {"kind": "inherits", "lookup": "unqualified", "path": ["Mixin"]},
                ],
                [{"kind": "calls", "lookup": "member", "path": ["m"]}],
                [],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.rev"}),
        expected=frozenset(
            [("C", "Base", "inherits"), ("C", "Mixin", "inherits"), ("C.run", "Mixin.m", "calls")]
        ),
    )


def case_a_reversed_language_searches_a_later_base_before_an_earlier_bases_base() -> CallScenario:
    """Ruby's order: `include A` (which includes `X`) then `include B`, `X` and `B` defining `m`: C, B, A, X, so `B.m`."""
    return CallScenario(
        rows={
            "id": ["C", "C.run", "A", "X", "X.m", "B", "B.m"],
            "file_path": ["a.rev"] * 7,
            "line_start": [1, 2, 4, 6, 7, 9, 10],
            "kind": ["class", "method", "class", "class", "method", "class", "method"],
            "name": ["C", "run", "A", "X", "m", "B", "m"],
            "scope": ["", "C", "", "", "X", "", "B"],
            "language": ["reversed"] * 7,
            "module": [None] * 7,
            "names": [None] * 7,
            "language_hint": [None] * 7,
            "bindings": [None] * 7,
            "refs": [
                [
                    {"kind": "inherits", "lookup": "unqualified", "path": ["A"]},
                    {"kind": "inherits", "lookup": "unqualified", "path": ["B"]},
                ],
                [{"kind": "calls", "lookup": "member", "path": ["m"]}],
                [{"kind": "inherits", "lookup": "unqualified", "path": ["X"]}],
                [],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.rev"}),
        expected=frozenset(
            [
                ("A", "X", "inherits"),
                ("C", "A", "inherits"),
                ("C", "B", "inherits"),
                ("C.run", "B.m", "calls"),
            ]
        ),
    )


def case_a_nearest_language_takes_the_shallowest_holder() -> CallScenario:
    """Go's order: `C` embeds `A` then `B`, `A` embeds `X`, `X` and `B` defining `M`: `B.M`, at depth 1."""
    return CallScenario(
        rows={
            "id": ["C", "C.run", "A", "X", "X.m", "B", "B.m"],
            "file_path": ["a.near"] * 7,
            "line_start": [1, 2, 4, 6, 7, 9, 10],
            "kind": ["class", "method", "class", "class", "method", "class", "method"],
            "name": ["C", "run", "A", "X", "m", "B", "m"],
            "scope": ["", "C", "", "", "X", "", "B"],
            "language": ["nearest"] * 7,
            "module": [None] * 7,
            "names": [None] * 7,
            "language_hint": [None] * 7,
            "bindings": [None] * 7,
            "refs": [
                [
                    {"kind": "inherits", "lookup": "unqualified", "path": ["A"]},
                    {"kind": "inherits", "lookup": "unqualified", "path": ["B"]},
                ],
                [{"kind": "calls", "lookup": "member", "path": ["m"]}],
                [{"kind": "inherits", "lookup": "unqualified", "path": ["X"]}],
                [],
                [],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.near"}),
        expected=frozenset(
            [
                ("A", "X", "inherits"),
                ("C", "A", "inherits"),
                ("C", "B", "inherits"),
                ("C.run", "B.m", "calls"),
            ]
        ),
    )


def case_a_listed_member_path_reaches_the_member() -> CallScenario:
    """`import static util.Strings.shout` lists `Strings.shout`: `shout()` reaches the member of `Strings`."""
    return CallScenario(
        rows={
            "id": ["imp", "run", "Strings", "Strings.shout"],
            "file_path": ["a.dot", "a.dot", "util/Strings.dot", "util/Strings.dot"],
            "line_start": [1, 3, 1, 2],
            "kind": ["import", "function", "class", "method"],
            "name": ["import static util.Strings.shout", "run", "Strings", "shout"],
            "scope": ["", "", "", "Strings"],
            "language": ["dotted"] * 4,
            "module": ["util.Strings", None, None, None],
            "names": ["Strings.shout", None, None, None],
            "language_hint": ["", None, None, None],
            "bindings": [[{"name": "Strings.shout", "local": "shout"}], None, None, None],
            "refs": [
                None,
                [{"kind": "calls", "lookup": "unqualified", "path": ["shout"], "type": None}],
                [],
                [],
            ],
        },
        repo_files=frozenset({"a.dot", "util/Strings.dot"}),
        expected=frozenset({("run", "Strings.shout", "calls")}),
    )


def case_a_member_also_reaches_its_definition_outside_the_class() -> CallScenario:
    """`Order::check` declared in its class and defined outside it: a call reaches both."""
    return CallScenario(
        rows={
            "id": ["Order", "Order.check", "check_def", "run"],
            "file_path": ["order.code", "order.code", "check.code", "order.code"],
            "line_start": [1, 2, 1, 5],
            "kind": ["class", "method", "function", "function"],
            "name": ["Order", "check", "check", "run"],
            "scope": ["", "Order", "Order", ""],
            "language": ["code"] * 4,
            "module": [None] * 4,
            "names": [None] * 4,
            "language_hint": [None] * 4,
            "bindings": [None] * 4,
            "refs": [
                [],
                [],
                [],
                [
                    {
                        "kind": "calls",
                        "lookup": "unqualified",
                        "path": ["Order", "check"],
                        "type": None,
                    }
                ],
            ],
        },
        repo_files=frozenset({"order.code", "check.code"}),
        expected=frozenset({("run", "Order.check", "calls"), ("run", "check_def", "calls")}),
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
        rows={
            "id": ["Order", "Order.save", "make_order", "run"],
            "file_path": ["a.code"] * 4,
            "line_start": [1, 2, 4, 6],
            "kind": ["class", "method", "function", "function"],
            "name": ["Order", "save", "make_order", "run"],
            "scope": ["", "Order", "", ""],
            "language": ["code"] * 4,
            "module": [None] * 4,
            "names": [None] * 4,
            "language_hint": [None] * 4,
            "bindings": [None] * 4,
            "refs": [
                [],
                [],
                [{"kind": "calls", "lookup": "unqualified", "path": ["Order"], "type": None}],
                [
                    {
                        "kind": "calls",
                        "lookup": "unqualified",
                        "path": ["make_order"],
                        "type": None,
                    },
                    {
                        "kind": "calls",
                        "lookup": "type",
                        "path": ["order", "save"],
                        "type": ["make_order"],
                    },
                ],
            ],
        },
        repo_files=frozenset({"a.code"}),
        expected=frozenset(
            {
                ("make_order", "Order", "calls"),
                ("run", "make_order", "calls"),
                ("run", "Order.save", "calls"),
            }
        ),
    )
