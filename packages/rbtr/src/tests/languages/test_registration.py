"""Tests for language plugin registration properties.

One parametrised test covering all languages. No pytest-cases
needed — flat value table.
"""

from __future__ import annotations

import pytest

from rbtr.languages import LanguageManager


@pytest.mark.parametrize(
    (
        "lang",
        "extensions",
        "grammar_module",
        "grammar_entry",
        "has_query",
        "has_extractor",
        "scope_types",
        "filenames",
    ),
    [
        (
            "python",
            frozenset({".py", ".pyi"}),
            "tree_sitter_python",
            "language",
            True,
            True,
            frozenset({"class_definition"}),
            frozenset(),
        ),
        (
            "javascript",
            frozenset({".js", ".jsx", ".mjs"}),
            "tree_sitter_javascript",
            "language",
            True,
            True,
            frozenset({"class_declaration"}),
            frozenset(),
        ),
        (
            "typescript",
            frozenset({".ts", ".tsx"}),
            "tree_sitter_typescript",
            "language_typescript",
            True,
            True,
            frozenset({"class_declaration"}),
            frozenset(),
        ),
        (
            "go",
            frozenset({".go"}),
            "tree_sitter_go",
            "language",
            True,
            True,
            frozenset({"type_spec"}),
            frozenset(),
        ),
        (
            "rust",
            frozenset({".rs"}),
            "tree_sitter_rust",
            "language",
            True,
            True,
            frozenset({"impl_item", "struct_item"}),
            frozenset(),
        ),
        (
            "java",
            frozenset({".java"}),
            "tree_sitter_java",
            "language",
            True,
            True,
            frozenset({"class_declaration"}),
            frozenset(),
        ),
        (
            "bash",
            frozenset({".sh", ".bash", ".zsh"}),
            "tree_sitter_bash",
            "language",
            True,
            False,
            frozenset(),
            frozenset({"Makefile", "Dockerfile", "Bashrc", ".bashrc", ".bash_profile", ".zshrc"}),
        ),
        (
            "c",
            frozenset({".c", ".h"}),
            "tree_sitter_c",
            "language",
            True,
            True,
            frozenset(),
            frozenset(),
        ),
        (
            "cpp",
            frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hxx"}),
            "tree_sitter_cpp",
            "language",
            True,
            True,
            frozenset({"class_specifier", "struct_specifier"}),
            frozenset(),
        ),
        (
            "ruby",
            frozenset({".rb"}),
            "tree_sitter_ruby",
            "language",
            True,
            True,
            frozenset({"class", "module"}),
            frozenset(),
        ),
    ],
    ids=[
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "bash",
        "c",
        "cpp",
        "ruby",
    ],
)
def test_registration_properties(
    lang: str,
    extensions: frozenset[str],
    grammar_module: str,
    grammar_entry: str,
    has_query: bool,
    has_extractor: bool,
    scope_types: frozenset[str],
    filenames: frozenset[str],
    language_manager: LanguageManager,
) -> None:
    """Every language plugin registers with the expected properties."""
    reg = language_manager.get_registration(lang)
    assert reg is not None, f"no registration for {lang}"
    assert reg.id == lang
    assert reg.extensions == extensions
    assert reg.grammar_module == grammar_module
    assert reg.grammar_entry == grammar_entry
    assert (reg.query is not None) == has_query
    assert (reg.import_extractor is not None) == has_extractor
    assert reg.scope_types == scope_types
    assert reg.filenames == filenames
