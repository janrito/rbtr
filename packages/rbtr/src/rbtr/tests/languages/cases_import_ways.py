"""Chunk rows as edge inference reads them, and the edges they make.

Each case writes `EdgeChunkRow` columns for neutral languages the test
module's `resolutions` fixture defines: `code`, whose references import
what they name, and `prose` and `page`, whose references document it; a
reference to a `prose` section compares folded, to a `page` section as
written. No plugin is involved, so each case states one rule of how an
import reaches what it imports, whatever language wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImportWayScenario:
    """`EdgeChunkRow` columns, the repository's files, and the edges inferred.

    An edge is `(source id, target id, kind)`.
    """

    rows: dict[str, list[str | int | list[str] | None]]
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
            "language_hint": ["", None, None],
        },
        repo_files=frozenset({"app.code", "lib.code"}),
        expected=frozenset({("import", "f", "imports")}),
    )


def case_an_import_of_every_name_reaches_what_a_bare_import_reaches() -> ImportWayScenario:
    """`*` names every name: the import reaches the module's definitions, as a bare one does."""
    return ImportWayScenario(
        rows={
            "id": ["import", "f", "C"],
            "file_path": ["app.code", "lib.code", "lib.code"],
            "line_start": [1, 1, 3],
            "kind": ["import", "function", "class"],
            "name": ["from lib import *", "f", "C"],
            "language": ["code"] * 3,
            "module": ["lib.code", None, None],
            "names": ["*", None, None],
            "language_hint": ["", None, None],
        },
        repo_files=frozenset({"app.code", "lib.code"}),
        expected=frozenset({("import", "f", "imports"), ("import", "C", "imports")}),
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
            "language_hint": ["", None],
        },
        repo_files=frozenset({"lib.code"}),
        expected=frozenset(),
    )


def case_a_file_reaches_itself_only_by_naming_itself() -> ImportWayScenario:
    """A name reaches the importing file only when the import names that file.

    `from . import helpers` in `pkg/index.code` resolves, through the
    directory's stand-in, to `pkg/index.code` itself; it names the
    directory, not the file, so its own `helpers` is not reached.
    """
    return ImportWayScenario(
        rows={
            "id": ["import", "helpers_var"],
            "file_path": ["pkg/index.code", "pkg/index.code"],
            "line_start": [1, 3],
            "kind": ["import", "variable"],
            "name": ["from . import helpers", "helpers"],
            "language": ["code"] * 2,
            "module": ["./", None],
            "names": ["helpers", None],
            "language_hint": ["", None],
        },
        repo_files=frozenset({"pkg/index.code"}),
        expected=frozenset(),
    )


def case_a_link_reaches_a_section_whose_title_folds_to_it() -> ImportWayScenario:
    """A link reaches a section of a folding language whose title equals it folded.

    Folded, case, accents and punctuation do not count, so each
    renderer's anchor for the heading reaches it; a name that folds to
    another title reaches that one instead.
    """
    return ImportWayScenario(
        rows={
            "id": ["hyphenated", "underscored", "accented", "getting_started", "cafe", "other"],
            "file_path": [
                "readme.prose",
                "readme.prose",
                "readme.prose",
                "guide.prose",
                "guide.prose",
                "guide.prose",
            ],
            "line_start": [1, 2, 3, 1, 5, 9],
            "kind": ["import", "import", "import", "doc_section", "doc_section", "doc_section"],
            "name": [
                "guide.prose#getting-started",
                "guide.prose#getting_started",
                "guide.prose#café----au-lait",
                "Getting Started",
                "Café -- au lait",
                "Other",
            ],
            "language": ["prose"] * 6,
            "module": ["guide.prose", "guide.prose", "guide.prose", None, None, None],
            "names": ["getting-started", "getting_started", "café----au-lait", None, None, None],
            "language_hint": ["", "", "", None, None, None],
        },
        repo_files=frozenset({"readme.prose", "guide.prose"}),
        expected=frozenset(
            {
                ("hyphenated", "getting_started", "documents"),
                ("underscored", "getting_started", "documents"),
                ("accented", "cafe", "documents"),
            }
        ),
    )


def case_a_link_reaches_a_section_of_an_exact_language_only_as_written() -> ImportWayScenario:
    """A link reaches a section of a language that does not fold only by its name as written."""
    return ImportWayScenario(
        rows={
            "id": ["slug", "id", "getting_started", "usage"],
            "file_path": ["readme.prose", "readme.prose", "site.page", "site.page"],
            "line_start": [1, 2, 1, 5],
            "kind": ["import", "import", "doc_section", "doc_section"],
            "name": ["site.page#getting-started", "site.page#usage", "Getting Started", "usage"],
            "language": ["prose", "prose", "page", "page"],
            "module": ["site.page", "site.page", None, None],
            "names": ["getting-started", "usage", None, None],
            "language_hint": ["", "", None, None],
        },
        repo_files=frozenset({"readme.prose", "site.page"}),
        expected=frozenset({("id", "usage", "documents")}),
    )


def case_a_link_reaches_a_section_in_its_own_file() -> ImportWayScenario:
    """A link whose module is its own file reaches a section of that file."""
    return ImportWayScenario(
        rows={
            "id": ["link", "usage"],
            "file_path": ["guide.prose", "guide.prose"],
            "line_start": [1, 5],
            "kind": ["import", "doc_section"],
            "name": ["#usage", "Usage"],
            "language": ["prose"] * 2,
            "module": ["guide.prose", None],
            "names": ["usage", None],
            "language_hint": ["", None],
        },
        repo_files=frozenset({"guide.prose"}),
        expected=frozenset({("link", "usage", "documents")}),
    )


def case_prose_naming_a_symbol_never_reaches_a_section() -> ImportWayScenario:
    """Prose naming a symbol reaches no section of that name, even when no symbol has it."""
    return ImportWayScenario(
        rows={
            "id": ["mention", "intro"],
            "file_path": ["readme.prose", "guide.prose"],
            "line_start": [1, 1],
            "kind": ["import", "doc_section"],
            "name": [":func:`intro`", "intro"],
            "language": ["prose", "prose"],
            "module": ["", None],
            "names": ["intro", None],
            "language_hint": ["", None],
        },
        repo_files=frozenset({"readme.prose", "guide.prose"}),
        expected=frozenset(),
    )


def case_a_reference_hinted_to_a_language_documents_its_first_section_of_that_name() -> (
    ImportWayScenario
):
    """A reference naming something and no file looks in the language its hint names.

    Hinted `prose` (an rST `:ref:`), it documents the first `prose`
    section answering to the name, by file then line, compared as `prose`
    folds; a `code` definition of that name is not where it looks.
    """
    return ImportWayScenario(
        rows={
            "id": ["ref", "intro_function", "intro_in_b", "intro_in_a"],
            "file_path": ["index.prose", "a.code", "b.prose", "a.prose"],
            "line_start": [1, 1, 1, 9],
            "kind": ["import", "function", "doc_section", "doc_section"],
            "name": [":ref:`intro`", "intro", "intro", "Intro"],
            "language": ["prose", "code", "prose", "prose"],
            "module": ["", None, None, None],
            "names": ["intro", None, None, None],
            "language_hint": ["prose", None, None, None],
        },
        repo_files=frozenset({"index.prose", "a.code", "a.prose", "b.prose"}),
        expected=frozenset({("ref", "intro_in_a", "documents")}),
    )


def case_a_reference_hinted_to_code_documents_its_definition() -> ImportWayScenario:
    """Hinted `code` (an rST `:func:`), a reference documents the `code` definition, not a section."""
    return ImportWayScenario(
        rows={
            "id": ["func", "intro_section", "intro_function"],
            "file_path": ["index.prose", "a.prose", "b.code"],
            "line_start": [1, 1, 1],
            "kind": ["import", "doc_section", "function"],
            "name": [":func:`intro`", "intro", "intro"],
            "language": ["prose", "prose", "code"],
            "module": ["", None, None],
            "names": ["intro", None, None],
            "language_hint": ["code", None, None],
        },
        repo_files=frozenset({"index.prose", "a.prose", "b.code"}),
        expected=frozenset({("func", "intro_function", "documents")}),
    )


def case_a_hinted_section_survives_another_language_in_the_same_file() -> ImportWayScenario:
    """An earlier section in another embedded language cannot hide the hinted section."""
    return ImportWayScenario(
        rows={
            "id": ["ref", "earlier", "wanted"],
            "file_path": ["index.prose", "mixed.host", "mixed.host"],
            "line_start": [1, 1, 9],
            "kind": ["import", "doc_section", "doc_section"],
            "name": ["reference Intro", "Intro", "Intro"],
            "language": ["prose", "code", "page"],
            "module": ["", None, None],
            "names": ["Intro", None, None],
            "language_hint": ["page", None, None],
        },
        repo_files=frozenset({"index.prose", "mixed.host"}),
        expected=frozenset({("ref", "wanted", "documents")}),
    )


def case_a_hinted_function_survives_another_language_in_the_same_file() -> ImportWayScenario:
    """An earlier definition in another embedded language cannot hide the hinted definition."""
    return ImportWayScenario(
        rows={
            "id": ["ref", "earlier", "wanted"],
            "file_path": ["index.prose", "mixed.host", "mixed.host"],
            "line_start": [1, 1, 9],
            "kind": ["import", "function", "function"],
            "name": ["reference intro", "intro", "intro"],
            "language": ["prose", "page", "code"],
            "module": ["", None, None],
            "names": ["intro", None, None],
            "language_hint": ["code", None, None],
        },
        repo_files=frozenset({"index.prose", "mixed.host"}),
        expected=frozenset({("ref", "wanted", "documents")}),
    )
