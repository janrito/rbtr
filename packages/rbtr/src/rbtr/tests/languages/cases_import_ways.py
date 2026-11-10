"""Chunk rows as edge inference reads them, and the edges they make.

Each case writes `EdgeChunkRow` columns for two neutral languages the
test module's `import_rules` fixture defines: `code`, whose references
import what they name, and `prose`, whose references document it. No
plugin is involved, so each case states one rule of how an import
reaches what it imports, whatever language wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImportWayScenario:
    """`EdgeChunkRow` columns, the repository's files, and the edges inferred.

    An edge is `(source id, target id, kind)`.
    """

    rows: dict[str, list[str | int | None]]
    repo_files: frozenset[str]
    expected: frozenset[tuple[str, str, str]]


def case_a_bare_import_brings_in_definitions() -> ImportWayScenario:
    """A bare import reaches the module's definitions, not its imports, comments or fragments."""
    return ImportWayScenario(
        rows={
            "id": ["import", "function", "class", "variable", "its_import", "comment", "fragment"],
            "file_path": ["app.code", *["lib.code"] * 6],
            "line_start": [1, 1, 3, 5, 7, 9, 11],
            "kind": ["import", "function", "class", "variable", "import", "comment", "fragment"],
            "name": ["import lib", "f", "C", "v", "import other", "", ""],
            "language": ["code"] * 7,
            "module": ["lib.code", None, None, None, "other.code", None, None],
            "names": ["", None, None, None, "", None, None],
            "dots": ["", None, None, None, "", None, None],
            "language_hint": ["", None, None, None, "", None, None],
        },
        repo_files=frozenset({"app.code", "lib.code"}),
        expected=frozenset(
            {
                ("import", "function", "imports"),
                ("import", "class", "imports"),
                ("import", "variable", "imports"),
            }
        ),
    )


def case_a_named_import_reaches_each_name_it_finds() -> ImportWayScenario:
    """A named import reaches each named symbol in the module; a name it does not hold reaches nothing."""
    return ImportWayScenario(
        rows={
            "id": ["import", "f", "g"],
            "file_path": ["app.code", "lib.code", "lib.code"],
            "line_start": [1, 1, 3],
            "kind": ["import", "function", "function"],
            "name": ["from lib import f, missing", "f", "g"],
            "language": ["code"] * 3,
            "module": ["lib.code", None, None],
            "names": ["f,missing", None, None],
            "dots": ["", None, None],
            "language_hint": ["", None, None],
        },
        repo_files=frozenset({"app.code", "lib.code"}),
        expected=frozenset({("import", "f", "imports")}),
    )


def case_a_name_no_symbol_holds_reaches_a_section_of_that_name() -> ImportWayScenario:
    """A name the module's symbols lack reaches its section of that name, and prose documents it."""
    return ImportWayScenario(
        rows={
            "id": ["link", "intro", "usage"],
            "file_path": ["readme.prose", "guide.prose", "guide.prose"],
            "line_start": [1, 1, 5],
            "kind": ["import", "doc_section", "doc_section"],
            "name": ["guide.prose#Usage", "Intro", "Usage"],
            "language": ["prose"] * 3,
            "module": ["guide.prose", None, None],
            "names": ["Usage", None, None],
            "dots": ["", None, None],
            "language_hint": ["", None, None],
        },
        repo_files=frozenset({"readme.prose", "guide.prose"}),
        expected=frozenset({("link", "usage", "documents")}),
    )


def case_prose_naming_a_symbol_documents_the_first_of_that_name() -> ImportWayScenario:
    """Prose naming a symbol and no file documents the first symbol of that name, by file and line."""
    return ImportWayScenario(
        rows={
            "id": ["mention", "f_in_b", "f_in_a"],
            "file_path": ["readme.prose", "b.code", "a.code"],
            "line_start": [1, 1, 9],
            "kind": ["import", "function", "function"],
            "name": [":func:`f`", "f", "f"],
            "language": ["prose", "code", "code"],
            "module": ["", None, None],
            "names": ["f", None, None],
            "dots": ["", None, None],
            "language_hint": ["", None, None],
        },
        repo_files=frozenset({"readme.prose", "a.code", "b.code"}),
        expected=frozenset({("mention", "f_in_a", "documents")}),
    )


def case_an_import_with_no_module_reaches_files_its_words_name() -> ImportWayScenario:
    """An import its extractor recorded nothing for reaches the first symbol of each file a word names.

    No word reaches a file whose stem is shorter than three characters
    (`io`), nor a directory's stand-in (`index`), which names its
    directory rather than a module.
    """
    return ImportWayScenario(
        rows={
            "id": ["import", "helper", "later_helper", "io_read", "index_main"],
            "file_path": ["app.code", "helpers.code", "helpers.code", "io.code", "pkg/index.code"],
            "line_start": [1, 1, 5, 1, 1],
            "kind": ["import", "function", "function", "function", "function"],
            "name": ["use helpers io index", "help", "assist", "read", "main"],
            "language": ["code"] * 5,
            "module": ["", None, None, None, None],
            "names": ["", None, None, None, None],
            "dots": ["", None, None, None, None],
            "language_hint": ["", None, None, None, None],
        },
        repo_files=frozenset({"app.code", "helpers.code", "io.code", "pkg/index.code"}),
        expected=frozenset({("import", "helper", "imports")}),
    )


def case_a_file_does_not_import_itself() -> ImportWayScenario:
    """An import whose module names its own file reaches nothing."""
    return ImportWayScenario(
        rows={
            "id": ["import", "f"],
            "file_path": ["lib.code", "lib.code"],
            "line_start": [1, 3],
            "kind": ["import", "function"],
            "name": ["import lib", "f"],
            "language": ["code"] * 2,
            "module": ["lib.code", None],
            "names": ["", None],
            "dots": ["", None],
            "language_hint": ["", None],
        },
        repo_files=frozenset({"lib.code"}),
        expected=frozenset(),
    )
