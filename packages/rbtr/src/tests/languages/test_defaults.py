"""Tests for the defaults plugin.

Verifies that detection-only and grammar-only languages are
registered correctly and behave as expected.
"""

from __future__ import annotations

import pytest

from rbtr.languages.hookspec import DEFAULT_SCOPE_TYPES
from rbtr.languages.manager import get_manager

_mgr = get_manager()

# ── Expected registrations ───────────────────────────────────────────

# Languages with grammar modules (may or may not be installed).
_GRAMMAR_LANGUAGES: dict[str, tuple[frozenset[str], str]] = {
    "c_sharp": (frozenset({".cs"}), "tree_sitter_c_sharp"),
    "css": (frozenset({".css"}), "tree_sitter_css"),
    "hcl": (frozenset({".hcl", ".tf"}), "tree_sitter_hcl"),
    "html": (frozenset({".html", ".htm"}), "tree_sitter_html"),
    "json": (frozenset({".json"}), "tree_sitter_json"),
    "kotlin": (frozenset({".kt", ".kts"}), "tree_sitter_kotlin"),
    "scala": (frozenset({".scala", ".sc"}), "tree_sitter_scala"),
    "swift": (frozenset({".swift"}), "tree_sitter_swift"),
    "toml": (frozenset({".toml"}), "tree_sitter_toml"),
    "yaml": (frozenset({".yaml", ".yml"}), "tree_sitter_yaml"),
}

# Detection-only languages (no grammar module).
_DETECTION_ONLY: dict[str, frozenset[str]] = {
    "markdown": frozenset({".md"}),
    "rst": frozenset({".rst"}),
}


# ── Registration tests ───────────────────────────────────────────────


@pytest.mark.parametrize("lang_id", sorted(_GRAMMAR_LANGUAGES))
def test_grammar_language_registered(lang_id: str) -> None:
    reg = _mgr.get_registration(lang_id)
    assert reg is not None, f"{lang_id} not registered"


@pytest.mark.parametrize("lang_id", sorted(_GRAMMAR_LANGUAGES))
def test_grammar_language_extensions(lang_id: str) -> None:
    expected_exts, _ = _GRAMMAR_LANGUAGES[lang_id]
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.extensions == expected_exts


@pytest.mark.parametrize("lang_id", sorted(_GRAMMAR_LANGUAGES))
def test_grammar_language_module(lang_id: str) -> None:
    _, expected_module = _GRAMMAR_LANGUAGES[lang_id]
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.grammar_module == expected_module


@pytest.mark.parametrize("lang_id", sorted(_GRAMMAR_LANGUAGES))
def test_grammar_language_no_query(lang_id: str) -> None:
    """Default languages should not have queries."""
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.query is None


@pytest.mark.parametrize("lang_id", sorted(_GRAMMAR_LANGUAGES))
def test_grammar_language_no_import_extractor(lang_id: str) -> None:
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.import_extractor is None


@pytest.mark.parametrize("lang_id", sorted(_GRAMMAR_LANGUAGES))
def test_grammar_language_default_scope_types(lang_id: str) -> None:
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.scope_types == DEFAULT_SCOPE_TYPES


@pytest.mark.parametrize("lang_id", sorted(_DETECTION_ONLY))
def test_detection_only_registered(lang_id: str) -> None:
    reg = _mgr.get_registration(lang_id)
    assert reg is not None


@pytest.mark.parametrize("lang_id", sorted(_DETECTION_ONLY))
def test_detection_only_extensions(lang_id: str) -> None:
    expected_exts = _DETECTION_ONLY[lang_id]
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.extensions == expected_exts


@pytest.mark.parametrize("lang_id", sorted(_DETECTION_ONLY))
def test_detection_only_no_grammar(lang_id: str) -> None:
    reg = _mgr.get_registration(lang_id)
    assert reg is not None
    assert reg.grammar_module is None


# ── Detection via file extension ─────────────────────────────────────


_EXTENSION_SAMPLES: list[tuple[str, str]] = [
    ("code.c", "c"),
    ("header.h", "c"),
    ("app.cs", "c_sharp"),
    ("main.cpp", "cpp"),
    ("header.hpp", "cpp"),
    ("styles.css", "css"),
    ("infra.tf", "hcl"),
    ("config.hcl", "hcl"),
    ("index.html", "html"),
    ("page.htm", "html"),
    ("data.json", "json"),
    ("app.kt", "kotlin"),
    ("build.kts", "kotlin"),
    ("app.rb", "ruby"),
    ("build.scala", "scala"),
    ("script.sc", "scala"),
    ("app.swift", "swift"),
    ("config.toml", "toml"),
    ("config.yaml", "yaml"),
    ("ci.yml", "yaml"),
    ("readme.md", "markdown"),
    ("docs.rst", "rst"),
]


@pytest.mark.parametrize(("filename", "expected_id"), _EXTENSION_SAMPLES)
def test_detect_language(filename: str, expected_id: str) -> None:
    assert _mgr.detect_language(filename) == expected_id


# ── Grammar loading ──────────────────────────────────────────────────


def test_detection_only_grammar_returns_none() -> None:
    assert _mgr.load_grammar("markdown") is None
    assert _mgr.load_grammar("rst") is None


def test_base_grammars_load() -> None:
    """JSON, YAML, TOML are base deps — always available."""
    for lang_id in ("json", "yaml", "toml"):
        g = _mgr.load_grammar(lang_id)
        assert g is not None, f"base grammar {lang_id} failed to load"
