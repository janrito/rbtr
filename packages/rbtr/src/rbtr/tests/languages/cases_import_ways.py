"""Source files that import, link and refer, and what each reaches.

Each case is files in the installed Python, Markdown, rST and JavaScript
plugins, or in the test module's `page` (Markdown whose sections compare
as written), and, for each target (a file and the symbol `find-refs` is
asked for), the referrers that reach it: the referrer's file, its name and
the edge. Hints name real languages (`:func:` is Python, `:ref:` rST), so
these cases use the installed registrations. A rule a plugin's own import
scenarios hold (a bare or named import, an import with no module, a link
to a heading, a folded anchor) is not repeated here.
"""

from __future__ import annotations

from dataclasses import dataclass

from rbtr.testing.assertions import Referrer, Target


@dataclass(frozen=True)
class ImportWayScenario:
    """Files, and for each target `(path, symbol)` the referrers `(path, name, edge)` that reach it."""

    files: dict[str, str]
    reached: dict[Target, set[Referrer]]


def case_an_import_of_every_name_reaches_what_a_bare_import_reaches() -> ImportWayScenario:
    """`*` names every name: the import reaches the module's definitions, as a bare one does."""
    return ImportWayScenario(
        files={
            "app.py": "from lib import *\n",
            "lib.py": """\
def f():
    pass


class C:
    pass
""",
        },
        reached={
            ("lib.py", "f"): {("app.py", "from lib import *", "imports")},
            ("lib.py", "C"): {("app.py", "from lib import *", "imports")},
        },
    )


def case_a_file_does_not_import_itself() -> ImportWayScenario:
    """An import whose module names its own file reaches nothing."""
    return ImportWayScenario(
        files={
            "lib.py": """\
import lib


def f():
    pass
""",
        },
        reached={("lib.py", "f"): set()},
    )


def case_a_file_reaches_itself_only_by_naming_itself() -> ImportWayScenario:
    """A name reaches the importing file only when the import names that file."""
    return ImportWayScenario(
        files={
            "pkg/__init__.py": """\
from . import helpers

helpers = 1
""",
        },
        reached={("pkg/__init__.py", "helpers"): set()},
    )


def case_a_link_reaches_a_section_of_an_exact_language_only_as_written() -> ImportWayScenario:
    """A link reaches a section of a language that does not fold only by its name as written."""
    return ImportWayScenario(
        files={
            "readme.md": """\
[a](site.page#getting-started)
[b](site.page#usage)
""",
            "site.page": """\
# Getting Started

text

# usage

text
""",
        },
        reached={
            ("site.page", "Getting Started"): set(),
            ("site.page", "usage"): {("readme.md", "site.page#usage", "documents")},
        },
    )


def case_a_link_reaches_a_section_in_its_own_file() -> ImportWayScenario:
    """A link whose module is its own file reaches a section of that file."""
    return ImportWayScenario(
        files={
            "guide.md": """\
[see](#usage)

# Usage

text
""",
        },
        reached={("guide.md", "Usage"): {("guide.md", "#usage", "documents")}},
    )


def case_prose_naming_a_symbol_documents_the_first_of_that_name() -> ImportWayScenario:
    """Prose naming a symbol and no file documents the first symbol of that name, by file and line."""
    return ImportWayScenario(
        files={
            "a.py": """\
x = 1


def f():
    pass
""",
            "b.py": """\
def f():
    pass
""",
            "readme.rst": "See :func:`f`.\n",
        },
        reached={
            ("a.py", "f"): {("readme.rst", ":func:`f`", "documents")},
            ("b.py", "f"): set(),
        },
    )


def case_prose_naming_a_symbol_never_reaches_a_section() -> ImportWayScenario:
    """Prose naming a symbol reaches no section of that name, even when no symbol has it."""
    return ImportWayScenario(
        files={
            "guide.md": """\
# intro

text
""",
            "readme.rst": "See :func:`intro`.\n",
        },
        reached={("guide.md", "intro"): set()},
    )


def case_a_reference_hinted_to_a_language_documents_its_first_section_of_that_name() -> (
    ImportWayScenario
):
    """A reference naming something and no file looks in the language it names: `:ref:` is rST's, folded."""
    return ImportWayScenario(
        files={
            "a.py": """\
def intro():
    pass
""",
            "a.rst": """\
Other
=====

text

Intro
=====

text
""",
            "b.rst": """\
intro
=====

text
""",
            "index.rst": "See :ref:`intro`.\n",
        },
        reached={
            ("a.rst", "Intro"): {("index.rst", ":ref:`intro`", "documents")},
            ("b.rst", "intro"): set(),
            ("a.py", "intro"): set(),
        },
    )


def case_a_reference_hinted_to_code_documents_its_definition() -> ImportWayScenario:
    """Hinted Python (an rST `:func:`), a reference documents the definition, not a section."""
    return ImportWayScenario(
        files={
            "a.rst": """\
intro
=====

text
""",
            "b.py": """\
def intro():
    pass
""",
            "index.rst": "See :func:`intro`.\n",
        },
        reached={
            ("b.py", "intro"): {("index.rst", ":func:`intro`", "documents")},
            ("a.rst", "intro"): set(),
        },
    )


def case_a_hinted_section_survives_another_language_in_the_same_file() -> ImportWayScenario:
    """An earlier section of another embedded language cannot hide the hinted section: Markdown's `Top.Intro` before rST's `Guide.Intro`."""
    return ImportWayScenario(
        files={
            "mixed.md": """\
# Top

## Intro

text

```rst
Guide
=====

Intro
-----

text
```
""",
            "index.rst": "See :ref:`Intro`.\n",
        },
        reached={
            ("mixed.md", "Top.Intro"): set(),
            ("mixed.md", "Guide.Intro"): {("index.rst", ":ref:`Intro`", "documents")},
        },
    )


def case_a_hinted_function_survives_another_language_in_the_same_file() -> ImportWayScenario:
    """An earlier definition of another embedded language cannot hide the hinted one: JavaScript's `Other.intro` before Python's `Guide.intro`."""
    return ImportWayScenario(
        files={
            "mixed.md": """\
# Guide

```javascript
class Other {
  intro() {}
}
```

```python
class Guide:
    def intro(self):
        pass
```
""",
            "index.rst": "See :func:`intro`.\n",
        },
        reached={
            ("mixed.md", "Other.intro"): set(),
            ("mixed.md", "Guide.intro"): {("index.rst", ":func:`intro`", "documents")},
        },
    )
