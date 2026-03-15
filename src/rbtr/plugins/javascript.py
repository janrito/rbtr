"""JavaScript and TypeScript language plugins.

Registers two languages (one per tree-sitter grammar) sharing a
common import extractor.  The only query difference is the class
name node type: JS uses `identifier`, TS uses `type_identifier`.

Extracted import examples::

    "import { foo, bar } from './models'"
        → {"module": "models", "names": "foo,bar", "dots": "1"}

    "import React from 'react'"
        → {"module": "react", "names": "React"}

    "import * as utils from '../utils'"
        → {"module": "utils", "names": "utils", "dots": "2"}

    "import type { Config } from './config'"  (TS only)
        → {"module": "config", "names": "Config", "dots": "1"}

    "import './styles.css'"  (side-effect)
        → {"module": "styles", "dots": "1"}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.index.models import ImportMeta
from rbtr.plugins.hookspec import LanguageRegistration, hookimpl, parse_path_relative

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Queries ──────────────────────────────────────────────────────────

# Shared patterns — identical across JS and TS grammars.
_SHARED = """\
(function_declaration
  name: (identifier) @_fn_name) @function

(import_statement) @import

(lexical_declaration
  (variable_declarator
    name: (identifier) @_fn_name
    value: (arrow_function))) @function
"""

# JS uses `identifier` for class names, TS uses `type_identifier`.
_JS_QUERY = (
    """\
(class_declaration
  name: (identifier) @_cls_name) @class

"""
    + _SHARED
)

_TS_QUERY = (
    """\
(class_declaration
  name: (type_identifier) @_cls_name) @class

"""
    + _SHARED
)

# ── Import extractor (shared by JS and TS) ───────────────────────────


def extract_import_meta(node: Node) -> ImportMeta:
    """Extract import data from a JS/TS `import_statement` node.

    Walks the tree-sitter AST handling four import forms:

    **Named imports** — `import { foo, bar } from './models'`::

        >>> extract_import_meta(node)
        {"module": "models", "names": "foo,bar", "dots": "1"}

    **Default import** — `import React from 'react'`::

        >>> extract_import_meta(node)
        {"module": "react", "names": "React"}

    **Namespace import** — `import * as utils from '../utils'`::

        >>> extract_import_meta(node)
        {"module": "utils", "names": "utils", "dots": "2"}

    **Side-effect import** — `import './styles.css'`::

        >>> extract_import_meta(node)
        {"module": "styles", "dots": "1"}

    Relative specifiers (`./`, `../`) are parsed into `dots`
    via `parse_path_relative`.  File
    extensions are stripped to mirror bundler resolution behaviour.
    """
    meta: ImportMeta = {}
    names: list[str] = []
    raw_module: str | None = None

    for child in node.children:
        match child.type:
            case "string":
                frag = child.child_by_field_name("content")
                if frag is None:
                    for sc in child.children:
                        if sc.type == "string_fragment" and sc.text:
                            raw_module = sc.text.decode()
                            break
                elif frag.text:
                    raw_module = frag.text.decode()

            case "import_clause":
                for ic in child.children:
                    match ic.type:
                        case "identifier":
                            if ic.text:
                                names.append(ic.text.decode())
                        case "named_imports":
                            for spec in ic.children:
                                if spec.type == "import_specifier":
                                    for sid in spec.children:
                                        if sid.type == "identifier" and sid.text:
                                            names.append(sid.text.decode())
                                            break
                        case "namespace_import":
                            for ns in ic.children:
                                if ns.type == "identifier" and ns.text:
                                    names.append(ns.text.decode())
                                    break

    if raw_module is not None:
        dots, cleaned = parse_path_relative(raw_module)
        if dots:
            meta["dots"] = str(dots)
        # Strip file extensions so edges.py can match without guessing.
        if "." in cleaned:
            stem, _sep, _ext = cleaned.rpartition(".")
            if stem:
                cleaned = stem
        meta["module"] = cleaned

    if names:
        meta["names"] = ",".join(names)
    return meta


# ── Plugin ───────────────────────────────────────────────────────────


class JavaScriptPlugin:
    """JavaScript and TypeScript language support.

    Registers two languages with separate grammars but a shared
    import extractor.  TypeScript requires `grammar_entry=
    "language_typescript"` because the `tree_sitter_typescript`
    package exposes `language_typescript()` instead of the
    standard `language()`.

    Example registration output::

        [
            LanguageRegistration(id="javascript", ...),
            LanguageRegistration(id="typescript", grammar_entry="language_typescript", ...),
        ]
    """

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="javascript",
                extensions=frozenset({".js", ".jsx", ".mjs"}),
                grammar_module="tree_sitter_javascript",
                query=_JS_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class_declaration"}),
            ),
            LanguageRegistration(
                id="typescript",
                extensions=frozenset({".ts", ".tsx"}),
                grammar_module="tree_sitter_typescript",
                grammar_entry="language_typescript",
                query=_TS_QUERY,
                import_extractor=extract_import_meta,
                scope_types=frozenset({"class_declaration"}),
            ),
        ]
