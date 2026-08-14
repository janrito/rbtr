"""One import per language, in that language's own syntax.

Each case gives two files: one importing, one imported. The test
extracts both the way a build does — each file as its own language —
and asserts that edge inference reaches the imported file.

The scenarios use a nested directory on purpose. A reference resolved
from the repository root happens to work when both files sit at the
root, which hides the most common shape of all: a file naming its
sibling.

Cases marked `xfail` record a resolution rbtr does not yet reach. The
mark is strict, so closing a gap turns its case red and prompts the
mark's removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest_cases import case


@dataclass(frozen=True)
class ImportScenario:
    """A file importing another, in one language's syntax."""

    language: str
    importer: str
    importer_source: str
    target: str
    target_source: str


# ── Resolving a reference to a file ──────────────────────────────────


def case_python() -> ImportScenario:
    """Python resolves a dotted module to a package file."""
    return ImportScenario(
        language="python",
        importer="pkg/a.py",
        importer_source="from pkg.b import helper\n",
        target="pkg/b.py",
        target_source="""\
def helper():
    return 1
""",
    )


def case_javascript() -> ImportScenario:
    """JavaScript resolves a relative specifier with an extension."""
    return ImportScenario(
        language="javascript",
        importer="src/a.js",
        importer_source="import { helper } from './b.js';\n",
        target="src/b.js",
        target_source="export function helper() {}\n",
    )


def case_typescript() -> ImportScenario:
    """TypeScript resolves an extensionless relative specifier."""
    return ImportScenario(
        language="typescript",
        importer="src/a.ts",
        importer_source="import { helper } from './b';\n",
        target="src/b.ts",
        target_source="export function helper(): number {\n  return 1;\n}\n",
    )


def case_tsx() -> ImportScenario:
    """TSX resolves an extensionless relative specifier."""
    return ImportScenario(
        language="tsx",
        importer="src/a.tsx",
        importer_source="import { Helper } from './b';\n",
        target="src/b.tsx",
        target_source="export function Helper() {\n  return null;\n}\n",
    )


def case_rust() -> ImportScenario:
    """Rust resolves a crate-relative `use` path."""
    return ImportScenario(
        language="rust",
        importer="src/a.rs",
        importer_source="use crate::b::helper;\n",
        target="src/b.rs",
        target_source="pub fn helper() -> u32 {\n    1\n}\n",
    )


def case_java() -> ImportScenario:
    """Java resolves a package-qualified import."""
    return ImportScenario(
        language="java",
        importer="a/A.java",
        importer_source="""\
import a.B;

class A {}
""",
        target="a/B.java",
        target_source="class B {}\n",
    )


def case_c() -> ImportScenario:
    """C resolves a quoted `#include` to a sibling header."""
    return ImportScenario(
        language="c",
        importer="src/a.c",
        importer_source='#include "b.h"\n',
        target="src/b.h",
        target_source="int helper(void);\n",
    )


def case_cpp() -> ImportScenario:
    """C++ resolves a quoted `#include` to a sibling header."""
    return ImportScenario(
        language="cpp",
        importer="src/a.cpp",
        importer_source='#include "b.hpp"\n',
        target="src/b.hpp",
        target_source="int helper();\n",
    )


def case_bash() -> ImportScenario:
    """Bash resolves a sourced sibling script."""
    return ImportScenario(
        language="bash",
        importer="bin/a.sh",
        importer_source="source ./b.sh\n",
        target="bin/b.sh",
        target_source="helper() {\n  :\n}\n",
    )


def case_svelte() -> ImportScenario:
    """A Svelte component resolves the script block's import."""
    return ImportScenario(
        language="svelte",
        importer="src/A.svelte",
        importer_source="""\
<script>
  import { helper } from './b.js';
</script>
""",
        target="src/b.js",
        target_source="export function helper() {}\n",
    )


def case_vue() -> ImportScenario:
    """A Vue component resolves the script block's import."""
    return ImportScenario(
        language="vue",
        importer="src/A.vue",
        importer_source="""\
<script>
import { helper } from './b.js';
</script>
""",
        target="src/b.js",
        target_source="export function helper() {}\n",
    )


# ── Not resolving yet ────────────────────────────────────────────────
#
# Each mark is strict, so closing a gap turns its case red and prompts
# the mark's removal.


def case_css() -> ImportScenario:
    """CSS `@import` names a sibling stylesheet."""
    return ImportScenario(
        language="css",
        importer="css/a.css",
        importer_source='@import "b.css";\n',
        target="css/b.css",
        target_source=".x {\n  color: red;\n}\n",
    )


def case_scss() -> ImportScenario:
    """SCSS `@use` names a sibling stylesheet."""
    return ImportScenario(
        language="scss",
        importer="scss/a.scss",
        importer_source='@use "b";\n',
        target="scss/b.scss",
        target_source="$colour: red;\n",
    )


def case_less() -> ImportScenario:
    """Less `@import` names a sibling stylesheet."""
    return ImportScenario(
        language="less",
        importer="less/a.less",
        importer_source='@import "b.less";\n',
        target="less/b.less",
        target_source="@colour: red;\n",
    )


def case_html() -> ImportScenario:
    """An HTML `<script src>` names a sibling script."""
    return ImportScenario(
        language="html",
        importer="web/a.html",
        importer_source='<script src="b.js"></script>\n',
        target="web/b.js",
        target_source="export function helper() {}\n",
    )


def case_markdown() -> ImportScenario:
    """A Markdown link names a sibling document."""
    return ImportScenario(
        language="markdown",
        importer="docs/a.md",
        importer_source="See [the guide](b.md) for details.\n",
        target="docs/b.md",
        target_source="# Guide\n\nBody.\n",
    )


def case_rst() -> ImportScenario:
    """An rST `:doc:` role names a sibling document."""
    return ImportScenario(
        language="rst",
        importer="docs/a.rst",
        importer_source="""\
Title
=====

See :doc:`b` for details.
""",
        target="docs/b.rst",
        target_source="""\
Guide
=====

Body.
""",
    )


def case_ruby() -> ImportScenario:
    """Ruby resolves a `require_relative` sibling."""
    return ImportScenario(
        language="ruby",
        importer="lib/a.rb",
        importer_source="require_relative 'b'\n",
        target="lib/b.rb",
        target_source="def helper\n  1\nend\n",
    )


@case(
    marks=pytest.mark.xfail(
        reason="the module prefix declared in go.mod is not stripped", strict=True
    )
)
def case_go() -> ImportScenario:
    """Go resolves an import naming a package inside the module."""
    return ImportScenario(
        language="go",
        importer="a/main.go",
        importer_source="""\
package main

import "example.com/m/b"
""",
        target="b/b.go",
        target_source="package b\n\nfunc Helper() int {\n\treturn 1\n}\n",
    )


@case(
    marks=pytest.mark.xfail(
        reason="a module block's source is not extracted as an import", strict=True
    )
)
def case_hcl() -> ImportScenario:
    """Terraform resolves a module block's source directory."""
    return ImportScenario(
        language="hcl",
        importer="tf/a.tf",
        importer_source="""\
module "x" {
  source = "./b"
}
""",
        target="tf/b/main.tf",
        target_source='resource "null_resource" "y" {}\n',
    )
