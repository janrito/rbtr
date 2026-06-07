"""Scenarios for `IndexStore.diff_symbols`.

Each case returns a `DiffScenario`: the file set of a *base*
commit, the file set of a *head* commit, and the exact symbols
expected in each change bucket. The driver in
`test_diff_symbols.py` builds both commits, indexes them, runs
`diff_symbols`, and asserts the buckets match.

Symbols are identified by `(name, scope)`; module-level symbols
have scope `""`, methods have their class name as scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pytest_cases import case

# (name, scope)
SymbolId = tuple[str, str]


@dataclass(frozen=True)
class DiffScenario:
    """A base→head file change and the symbols it should surface."""

    base_files: dict[str, bytes]
    head_files: dict[str, bytes]
    expected_added: set[SymbolId] = field(default_factory=set)
    expected_modified: set[SymbolId] = field(default_factory=set)
    expected_removed: set[SymbolId] = field(default_factory=set)
    # Diff base against itself (head_files ignored).
    same_as_base: bool = False
    # Index head under its *tree* SHA, not its commit SHA — exercises
    # the worktree path, where a working tree is indexed by tree SHA.
    head_as_tree: bool = False


# A two-function module reused by several cases. Cases that must
# leave a symbol *unchanged* repeat its exact bytes so the content
# comparison sees no difference.
_UTILS_BASE = b"""\
def helper():
    return 42

def format_name(name):
    return name.strip()
"""


@case(tags=["diff"])
def case_added() -> DiffScenario:
    """A new function in an existing file → added; neighbours untouched."""
    head = b"""\
def helper():
    return 42

def format_name(name):
    return name.strip()

def new_func():
    return "new"
"""
    return DiffScenario(
        base_files={"utils.py": _UTILS_BASE},
        head_files={"utils.py": head},
        expected_added={("new_func", "")},
    )


@case(tags=["diff"])
def case_removed() -> DiffScenario:
    """Deleting one class → removed; the surviving class is untouched."""
    base = b"""\
class User:
    pass

class Order:
    pass
"""
    head = b"""\
class User:
    pass
"""
    return DiffScenario(
        base_files={"models.py": base},
        head_files={"models.py": head},
        expected_removed={("Order", "")},
    )


@case(tags=["diff"])
def case_modified_is_precise() -> DiffScenario:
    """Change one function body → only it is modified.

    Doubles as the precision guard: `format_name` is byte-identical
    across both sides, so exact-set equality proves the unchanged
    neighbour does not leak in.
    """
    head = b"""\
def helper():
    return 43

def format_name(name):
    return name.strip()
"""
    return DiffScenario(
        base_files={"utils.py": _UTILS_BASE},
        head_files={"utils.py": head},
        expected_modified={("helper", "")},
    )


@case(tags=["diff"])
def case_rename() -> DiffScenario:
    """A rename is a remove of the old name plus an add of the new."""
    head = b"""\
def helper2():
    return 42

def format_name(name):
    return name.strip()
"""
    return DiffScenario(
        base_files={"utils.py": _UTILS_BASE},
        head_files={"utils.py": head},
        expected_added={("helper2", "")},
        expected_removed={("helper", "")},
    )


@case(tags=["diff"])
def case_new_file() -> DiffScenario:
    """A wholly-new file contributes all its symbols as added."""
    service = b"""\
def serve():
    return True
"""
    return DiffScenario(
        base_files={"utils.py": _UTILS_BASE},
        head_files={"utils.py": _UTILS_BASE, "service.py": service},
        expected_added={("serve", "")},
    )


@case(tags=["diff"])
def case_removed_file() -> DiffScenario:
    """A wholly-removed file contributes all its symbols as removed."""
    main = b"""\
def run():
    return 1
"""
    return DiffScenario(
        base_files={"utils.py": _UTILS_BASE, "main.py": main},
        head_files={"utils.py": _UTILS_BASE},
        expected_removed={("run", "")},
    )


@case(tags=["diff"])
def case_no_change() -> DiffScenario:
    """Diffing a commit against itself yields nothing."""
    return DiffScenario(
        base_files={"utils.py": _UTILS_BASE},
        head_files={},
        same_as_base=True,
    )


@case(tags=["diff"])
def case_method_flags_class() -> DiffScenario:
    """Editing a method body also flags its enclosing class.

    The class chunk's content spans its body, so a method change
    alters both chunks. This pins the real behaviour.
    """
    base = b"""\
class Svc:
    def start(self):
        return 1
"""
    head = b"""\
class Svc:
    def start(self):
        return 2
"""
    return DiffScenario(
        base_files={"svc.py": base},
        head_files={"svc.py": head},
        expected_modified={("start", "Svc"), ("Svc", "")},
    )


@case(tags=["diff"])
def case_worktree() -> DiffScenario:
    """Head indexed by tree SHA (worktree path) diffs like a commit."""
    head = b"""\
def helper():
    return 43

def format_name(name):
    return name.strip()
"""
    return DiffScenario(
        base_files={"utils.py": _UTILS_BASE},
        head_files={"utils.py": head},
        expected_modified={("helper", "")},
        head_as_tree=True,
    )
