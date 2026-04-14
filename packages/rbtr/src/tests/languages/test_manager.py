"""Tests for the language plugin manager.

Tests the LanguageManager's detection, grammar loading, query
retrieval, and import extraction delegation.
"""

from __future__ import annotations

import pluggy
import pytest

from rbtr.languages import LanguageManager, get_manager
from rbtr.languages.hookspec import LanguageHookspec, LanguageRegistration, hookimpl

_mgr = get_manager()

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
        "java",
        "json",
        "yaml",
        "yml",
        "toml",
        "markdown",
        "nested-path",
    ],
)
def test_detect_language(path: str, expected: str) -> None:
    assert _mgr.detect_language(path) == expected


@pytest.mark.parametrize(
    "path",
    ["README", "data.xyz"],
    ids=["no-extension", "unknown-extension"],
)
def test_detect_unknown_returns_none(path: str) -> None:
    assert _mgr.detect_language(path) is None


def test_detect_filename_priority_over_extension() -> None:
    """Filenames are checked before extensions."""
    assert _mgr.detect_language("Makefile") == "bash"


# ── get_registration ─────────────────────────────────────────────────


def test_get_registration_exists() -> None:
    reg = _mgr.get_registration("python")
    assert reg is not None
    assert reg.id == "python"


def test_get_registration_none_for_unknown() -> None:
    assert _mgr.get_registration("nonexistent") is None


# ── load_grammar ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "bash", "json", "yaml", "toml"],
    ids=["python", "bash", "json", "yaml", "toml"],
)
def test_load_grammar_base(lang_id: str) -> None:
    assert _mgr.load_grammar(lang_id) is not None


@pytest.mark.parametrize(
    "lang_id",
    ["markdown", "nonexistent"],
    ids=["detection-only", "unknown"],
)
def test_load_grammar_returns_none(lang_id: str) -> None:
    assert _mgr.load_grammar(lang_id) is None


def test_load_grammar_cached() -> None:
    """Same object returned on repeated calls."""
    g1 = _mgr.load_grammar("python")
    g2 = _mgr.load_grammar("python")
    assert g1 is g2


# ── get_query ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "bash"],
)
def test_get_query_has_function(lang_id: str) -> None:
    q = _mgr.get_query(lang_id)
    assert q is not None
    assert "@function" in q


@pytest.mark.parametrize(
    "lang_id",
    ["json", "markdown", "nonexistent"],
    ids=["grammar-no-query", "detection-only", "unknown"],
)
def test_get_query_returns_none(lang_id: str) -> None:
    assert _mgr.get_query(lang_id) is None


def test_python_query_has_import_and_class() -> None:
    q = _mgr.get_query("python")
    assert q is not None
    assert "@import" in q
    assert "@class" in q


def test_bash_query_has_function_only() -> None:
    q = _mgr.get_query("bash")
    assert q is not None
    assert "@function" in q
    assert "@class" not in q
    assert "@import" not in q


# ── get_scope_types ──────────────────────────────────────────────────


def test_scope_types_python() -> None:
    assert "class_definition" in _mgr.get_scope_types("python")


@pytest.mark.parametrize(
    "lang_id",
    ["bash", "nonexistent"],
    ids=["no-scope-types", "unknown"],
)
def test_scope_types_empty(lang_id: str) -> None:
    assert _mgr.get_scope_types(lang_id) == frozenset()


# ── get_language ─────────────────────────────────────────────────────


def test_get_language_python() -> None:
    result = _mgr.get_language("app.py")
    assert result is not None
    lang_id, grammar = result
    assert lang_id == "python"
    assert grammar is not None


def test_get_language_bash() -> None:
    result = _mgr.get_language("script.sh")
    assert result is not None
    assert result[0] == "bash"


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("data.xyz", "unknown extension"),
        ("README.md", "detection-only, no grammar"),
    ],
    ids=["unknown", "detection-only"],
)
def test_get_language_returns_none(path: str, reason: str) -> None:
    assert _mgr.get_language(path) is None


# ── missing_grammar ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "bash"],
)
def test_missing_grammar_false_for_base(lang_id: str) -> None:
    assert not _mgr.missing_grammar(lang_id)


def test_missing_grammar_false_for_chunker_only() -> None:
    """Languages with a custom chunker but no grammar_module are not 'missing'."""
    assert not _mgr.missing_grammar("markdown")
    assert not _mgr.missing_grammar("rst")


# ── all_language_ids ─────────────────────────────────────────────────


def test_all_language_ids_not_empty() -> None:
    ids = _mgr.all_language_ids()
    assert len(ids) > 0


@pytest.mark.parametrize(
    "lang_id",
    [
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
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
def test_all_language_ids_contains(lang_id: str) -> None:
    assert lang_id in _mgr.all_language_ids()


# ── extract_import_meta delegation ───────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["json", "nonexistent"],
    ids=["no-extractor", "unknown-language"],
)
def test_extract_import_meta_returns_empty(lang_id: str) -> None:
    meta = _mgr.extract_import_meta(lang_id, None)  # type: ignore[arg-type]  # testing None node
    assert meta == {}


# ── Registrations with import extractors ─────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "ruby"],
)
def test_language_has_extractor(lang_id: str) -> None:
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.import_extractor is not None, f"missing extractor for {lang_id}"


@pytest.mark.parametrize(
    "lang_id",
    ["bash", "json", "yaml", "toml", "css", "html", "markdown"],
)
def test_language_has_no_extractor(lang_id: str) -> None:
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.import_extractor is None, f"unexpected extractor for {lang_id}"


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
