"""Tests for the language plugin manager.

Tests the LanguageManager's detection, grammar loading, query
retrieval, and import extraction delegation.
"""

from __future__ import annotations

import pluggy
import pytest
from tree_sitter import Query

from rbtr.languages import LanguageManager
from rbtr.languages.hookspec import LanguageHookspec, LanguageRegistration, hookimpl

# ── detect_language ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/app.py", "python"),
        ("stubs/foo.pyi", "python"),
        ("Makefile", "bash"),
        ("Dockerfile", "bash"),
        (".bashrc", "bash"),
        ("index.js", "javascript"),
        ("App.jsx", "javascript"),
        ("module.mjs", "javascript"),
        ("app.ts", "typescript"),
        ("Component.tsx", "typescript"),
        ("main.go", "go"),
        ("lib.rs", "rust"),
        ("schema.sql", "sql"),
        ("App.java", "java"),
        ("package.json", "json"),
        ("config.yaml", "yaml"),
        ("ci.yml", "yaml"),
        ("pyproject.toml", "toml"),
        ("README.md", "markdown"),
        ("src/rbtr/index/store.py", "python"),
    ],
    ids=[
        "py",
        "pyi",
        "makefile",
        "dockerfile",
        "bashrc",
        "js",
        "jsx",
        "mjs",
        "ts",
        "tsx",
        "go",
        "rust",
        "sql",
        "java",
        "json",
        "yaml",
        "yml",
        "toml",
        "markdown",
        "nested-path",
    ],
)
def test_detect_language(path: str, expected: str, language_manager: LanguageManager) -> None:
    assert language_manager.detect_language(path) == expected


@pytest.mark.parametrize(
    "path",
    ["README", "data.xyz"],
    ids=["no-extension", "unknown-extension"],
)
def test_detect_unknown_returns_none(path: str, language_manager: LanguageManager) -> None:
    assert language_manager.detect_language(path) is None


def test_detect_filename_priority_over_extension(language_manager: LanguageManager) -> None:
    """Filenames are checked before extensions."""
    assert language_manager.detect_language("Makefile") == "bash"


# ── get_registration ─────────────────────────────────────────────────


def test_get_registration_exists(language_manager: LanguageManager) -> None:
    reg = language_manager.get_registration("python")
    assert reg is not None
    assert reg.id == "python"


def test_get_registration_none_for_unknown(language_manager: LanguageManager) -> None:
    assert language_manager.get_registration("nonexistent") is None


# ── load_grammar ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "bash", "json", "yaml", "toml"],
    ids=["python", "bash", "json", "yaml", "toml"],
)
def test_load_grammar_base(lang_id: str, language_manager: LanguageManager) -> None:
    assert language_manager.load_grammar(lang_id) is not None


@pytest.mark.parametrize(
    "lang_id",
    ["nonexistent"],
    ids=["unknown"],
)
def test_load_grammar_returns_none(lang_id: str, language_manager: LanguageManager) -> None:
    assert language_manager.load_grammar(lang_id) is None


def test_load_grammar_cached(language_manager: LanguageManager) -> None:
    """Same object returned on repeated calls."""
    g1 = language_manager.load_grammar("python")
    g2 = language_manager.load_grammar("python")
    assert g1 is g2


# ── get_query ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "bash"],
)
def test_get_query_has_function(lang_id: str, language_manager: LanguageManager) -> None:
    q = language_manager.get_query(lang_id)
    assert q is not None
    assert "@function" in q


@pytest.mark.parametrize(
    "lang_id",
    ["markdown", "nonexistent"],
    ids=["chunker-no-query", "unknown"],
)
def test_get_query_returns_none(lang_id: str, language_manager: LanguageManager) -> None:
    assert language_manager.get_query(lang_id) is None


def test_python_query_has_import_and_class(language_manager: LanguageManager) -> None:
    q = language_manager.get_query("python")
    assert q is not None
    assert "@import" in q
    assert "@class" in q


def test_bash_query_has_function_and_import(language_manager: LanguageManager) -> None:
    q = language_manager.get_query("bash")
    assert q is not None
    assert "@function" in q
    assert "@import" in q
    assert "@class" not in q


def test_every_query_compiles_against_its_grammar(language_manager: LanguageManager) -> None:
    """Each registered query compiles — guards against grammar-version drift.

    Query strings reference grammar-specific node types; a grammar bump that
    renames a node makes the whole query fail to compile. Compiling each here
    turns that into one located failure rather than scattered index errors.
    """
    for lang_id in language_manager.all_language_ids():
        query_str = language_manager.get_query(lang_id)
        grammar = language_manager.load_grammar(lang_id)
        if query_str is None or grammar is None:
            continue
        Query(grammar, query_str)  # raises QueryError on an unknown node type


# ── get_scope_types ──────────────────────────────────────────────────


def test_scope_types_python(language_manager: LanguageManager) -> None:
    assert "class_definition" in language_manager.get_scope_types("python")


@pytest.mark.parametrize(
    "lang_id",
    ["bash", "nonexistent"],
    ids=["no-scope-types", "unknown"],
)
def test_scope_types_empty(lang_id: str, language_manager: LanguageManager) -> None:
    assert language_manager.get_scope_types(lang_id) == frozenset()


# ── get_language ─────────────────────────────────────────────────────


def test_get_language_python(language_manager: LanguageManager) -> None:
    result = language_manager.get_language("app.py")
    assert result is not None
    lang_id, grammar = result
    assert lang_id == "python"
    assert grammar is not None


def test_get_language_bash(language_manager: LanguageManager) -> None:
    result = language_manager.get_language("script.sh")
    assert result is not None
    assert result[0] == "bash"


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("data.xyz", "unknown extension"),
    ],
    ids=["unknown"],
)
def test_get_language_returns_none(
    path: str, reason: str, language_manager: LanguageManager
) -> None:
    assert language_manager.get_language(path) is None


# ── missing_grammar ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "bash"],
)
def test_missing_grammar_false_for_base(lang_id: str, language_manager: LanguageManager) -> None:
    assert not language_manager.missing_grammar(lang_id)


def test_missing_grammar_false_for_chunker_only(language_manager: LanguageManager) -> None:
    """Languages with a custom chunker but no grammar_module are not 'missing'."""
    assert not language_manager.missing_grammar("markdown")
    assert not language_manager.missing_grammar("rst")


# ── all_language_ids ─────────────────────────────────────────────────


def test_all_language_ids_not_empty(language_manager: LanguageManager) -> None:
    ids = language_manager.all_language_ids()
    assert len(ids) > 0


@pytest.mark.parametrize(
    "lang_id",
    [
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "sql",
        "java",
        "bash",
        "json",
        "yaml",
        "toml",
        "markdown",
        "css",
        "html",
    ],
)
def test_all_language_ids_contains(lang_id: str, language_manager: LanguageManager) -> None:
    assert lang_id in language_manager.all_language_ids()


# ── Duplicate detection ──────────────────────────────────────────────


def test_duplicate_id_in_single_plugin_raises() -> None:
    """A plugin returning two registrations with the same ID is an error."""

    class BadPlugin:
        @hookimpl
        def rbtr_register_languages(self) -> list[LanguageRegistration]:
            return [
                LanguageRegistration(id="dup", extensions=frozenset({".dup1"})),
                LanguageRegistration(id="dup", extensions=frozenset({".dup2"})),
            ]

    mgr = LanguageManager.__new__(LanguageManager)
    mgr._registrations = {}
    mgr._ext_map = {}
    mgr._filename_map = {}
    mgr._grammar_cache = {}

    mgr._pm = pluggy.PluginManager("rbtr")

    # Avoid circular import — import spec directly.

    mgr._pm.add_hookspecs(LanguageHookspec)
    mgr._pm.register(BadPlugin())

    with pytest.raises(ValueError, match="duplicate language id"):
        mgr._collect()
