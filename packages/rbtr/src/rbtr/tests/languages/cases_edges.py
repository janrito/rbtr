"""Repositories, and the edges a build of each infers.

Each case gives a repository's files and the edges `render_edges` reads
back from its build, as `source -> target [kind]` lines naming each end
by `path::name`. The one-per-language cases import a file in that
language's own syntax; the others pin a rule of how imports resolve.

The language cases use nested directories on purpose. A reference
resolved from the repository root happens to work when both files sit
at the root, which hides the most common shape of all: a file naming
its sibling.

A language that cannot yet follow its own syntax belongs here with a
strict `xfail` naming what is missing, so closing the gap turns the case
red and prompts the mark's removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class EdgeScenario:
    """A repository's files, by path, and the edges a build of them infers."""

    files: dict[str, bytes]
    expected: list[str]


def case_python_named_imports() -> EdgeScenario:
    """A named import reaches each name: a function, a class, variables, and a relative module's class."""
    return EdgeScenario(
        files={
            "pkg/a.py": b"""\
from pkg.b import helper, Config
from pkg.config import settings, high
from .models import User
""",
            "pkg/b.py": b"""\
def helper():
    return 1


class Config:
    pass
""",
            "pkg/config.py": b"""\
settings = {}
low, high = 1, 9
""",
            "pkg/models.py": b"""\
class User:
    pass
""",
        },
        expected=[
            "pkg/a.py::from .models import User -> pkg/models.py::User [imports]",
            "pkg/a.py::from pkg.b import helper, Config -> pkg/b.py::Config [imports]",
            "pkg/a.py::from pkg.b import helper, Config -> pkg/b.py::helper [imports]",
            "pkg/a.py::from pkg.config import settings, high -> pkg/config.py::high [imports]",
            "pkg/a.py::from pkg.config import settings, high -> pkg/config.py::settings [imports]",
        ],
    )


def case_python_bare_import() -> EdgeScenario:
    """A bare import reaches a module's definitions, not its imports or the code no query names."""
    return EdgeScenario(
        files={
            "pkg/a.py": b"""\
import pkg.models
""",
            "pkg/models.py": b"""\
import os


class User:
    pass


def load():
    return User()


load()
""",
        },
        expected=[
            "pkg/a.py::import pkg.models -> pkg/models.py::User [imports]",
            "pkg/a.py::import pkg.models -> pkg/models.py::load [imports]",
        ],
    )


def case_python_suffix_in_a_monorepo() -> EdgeScenario:
    """An absolute import resolves across a `packages/*/src` layout by its full path suffix.

    `other`'s `helpers.py` shares the file name but not the suffix, so it
    is not reached.
    """
    return EdgeScenario(
        files={
            "packages/app/src/app/main.py": b"""\
from core.models import Model
from core.helpers import assist
""",
            "packages/core/src/core/models.py": b"""\
class Model:
    pass
""",
            "packages/core/src/core/helpers.py": b"""\
def assist():
    return 1
""",
            "packages/other/src/other/helpers.py": b"""\
def assist():
    return 2
""",
        },
        expected=[
            "packages/app/src/app/main.py::from core.helpers import assist -> packages/core/src/core/helpers.py::assist [imports]",
            "packages/app/src/app/main.py::from core.models import Model -> packages/core/src/core/models.py::Model [imports]",
        ],
    )


def case_python_suffix_in_two_packages() -> EdgeScenario:
    """A full path suffix two packages share is ambiguous, so it reaches neither."""
    return EdgeScenario(
        files={
            "packages/app/src/app/main.py": b"""\
from core.models import Model
""",
            "packages/a/src/core/models.py": b"""\
class Model:
    pass
""",
            "packages/b/src/core/models.py": b"""\
class Model:
    pass
""",
        },
        expected=[],
    )


def case_python_imports_that_reach_nothing() -> EdgeScenario:
    """Imports whose module the repository does not hold reach nothing.

    - `os` is not a repository file.
    - `from . import utils` names a package with no `__init__.py`: Python
      binds the package, not the modules beside it.
    - `Missing` is not defined in `pkg/models.py`.
    - Five dots climb past the repository root.
    - `helpers` is one segment, which never matches a nested file by
      suffix.
    """
    return EdgeScenario(
        files={
            "pkg/app.py": b"""\
import os
from . import utils
from .models import Missing
from ..... import deep
import helpers
""",
            "pkg/utils.py": b"""\
def util():
    return 1
""",
            "pkg/models.py": b"""\
class User:
    pass
""",
            "deep/nested/helpers.py": b"""\
def assist():
    return 1
""",
        },
        expected=[],
    )


def case_python_import_with_no_module() -> EdgeScenario:
    """An import its extractor records nothing for reaches files its words name.

    `from __future__ import annotations` names no repository module; its
    word `annotations` is the stem of `pkg/annotations.py`, whose first
    symbol it reaches.
    """
    return EdgeScenario(
        files={
            "pkg/a.py": b"""\
from __future__ import annotations
""",
            "pkg/annotations.py": b"""\
def note():
    return 1


def other():
    return 2
""",
        },
        expected=[
            "pkg/a.py::from __future__ import annotations -> pkg/annotations.py::note [imports]",
        ],
    )


def case_javascript_imports_a_stylesheet() -> EdgeScenario:
    """A JavaScript import of a stylesheet reaches the stylesheet's rules."""
    return EdgeScenario(
        files={
            "src/a.js": b"""\
import './styles.css';
""",
            "src/styles.css": b"""\
.banner {
  color: red;
}
""",
        },
        expected=[
            "src/a.js::import './styles.css'; -> src/styles.css::.banner [imports]",
        ],
    )


def case_html_scripts_and_a_stylesheet() -> EdgeScenario:
    """HTML resolves `<script src>` beside it and above it, and `<link href>` to a stylesheet."""
    return EdgeScenario(
        files={
            "web/index.html": b"""\
<script src="./app.js"></script>
<script src="../lib/core.js"></script>
<link rel="stylesheet" href="styles.css">
""",
            "web/app.js": b"""\
export function start() {}
""",
            "lib/core.js": b"""\
export function boot() {}

export function halt() {}
""",
            "web/styles.css": b"""\
.banner {
  color: red;
}
""",
        },
        expected=[
            'web/index.html::<link rel="stylesheet" href="styles.css"> -> web/styles.css::.banner [imports]',
            'web/index.html::<script src="../lib/core.js"></script> -> lib/core.js::boot [imports]',
            'web/index.html::<script src="../lib/core.js"></script> -> lib/core.js::halt [imports]',
            'web/index.html::<script src="./app.js"></script> -> web/app.js::start [imports]',
        ],
    )


def case_markdown_links() -> EdgeScenario:
    """A Markdown link documents what it names: a code file's symbols, or a document's sections.

    A name mentioned without a link documents nothing.
    """
    return EdgeScenario(
        files={
            "docs/a.md": b"""\
# Overview

See [the guide](b.md).

The [loader](../src/loader.py) reads files; `do_stuff` is only mentioned.
""",
            "docs/b.md": b"""\
# Guide

Body.

## Details

More.
""",
            "src/loader.py": b"""\
def load():
    return 1


def do_stuff():
    return 2
""",
        },
        expected=[
            "docs/a.md::../src/loader.py -> src/loader.py::do_stuff [documents]",
            "docs/a.md::../src/loader.py -> src/loader.py::load [documents]",
            "docs/a.md::b.md -> docs/b.md::Details [documents]",
            "docs/a.md::b.md -> docs/b.md::Guide [documents]",
        ],
    )


@pytest.mark.xfail(
    reason="a link names a heading by its anchor (`#details`), a section by its text (`Details`)",
    strict=True,
)
def case_markdown_link_to_a_heading() -> EdgeScenario:
    """A Markdown link to a heading's anchor documents that section alone."""
    return EdgeScenario(
        files={
            "docs/a.md": b"""\
See [the details](b.md#details).
""",
            "docs/b.md": b"""\
# Guide

Body.

## Details

More.
""",
        },
        expected=["docs/a.md::b.md#details -> docs/b.md::Details [documents]"],
    )
