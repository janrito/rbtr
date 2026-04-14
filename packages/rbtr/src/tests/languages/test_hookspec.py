"""Tests for hookspec utilities and LanguageRegistration."""

from __future__ import annotations

import pytest
from tree_sitter import Language, Node, Parser

from rbtr.index.models import ImportMeta
from rbtr.languages.hookspec import (
    DEFAULT_SCOPE_TYPES,
    LanguageRegistration,
    collect_scoped_path,
    parse_path_relative,
)
from rbtr.languages.manager import get_manager

# ── parse_path_relative ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected_dots", "expected_cleaned"),
    [
        ("./models", 1, "models"),
        ("../utils", 2, "utils"),
        ("../../shared/helpers", 3, "shared/helpers"),
        ("../../../root", 4, "root"),
        ("./a/b/c", 1, "a/b/c"),
        ("./styles.css", 1, "styles.css"),
        ("react", 0, "react"),
        ("@testing-library/react", 0, "@testing-library/react"),
        ("", 0, ""),
        (".", 0, "."),
    ],
    ids=[
        "current-dir",
        "parent-dir",
        "grandparent-dir",
        "triple-parent",
        "current-dir-nested",
        "current-with-extension",
        "absolute-package",
        "scoped-package",
        "empty-string",
        "single-dot",
    ],
)
def test_parse_path_relative(path: str, expected_dots: int, expected_cleaned: str) -> None:
    dots, cleaned = parse_path_relative(path)
    assert dots == expected_dots
    assert cleaned == expected_cleaned


# ── collect_scoped_path (via Rust grammar) ───────────────────────────

_mgr = get_manager()
_has_rust = _mgr.load_grammar("rust") is not None
skip_no_rust = pytest.mark.skipif(not _has_rust, reason="tree-sitter-rust not installed")


def _collect_from_use(code: str) -> list[str]:
    """Parse a Rust `use` declaration and collect scoped path parts."""
    import tree_sitter_rust  # optional grammar; guarded by skip_no_rust

    lang = Language(tree_sitter_rust.language())
    parser = Parser(lang)
    tree = parser.parse(code.encode())
    use_decl = tree.root_node.children[0]
    for child in use_decl.children:
        if child.type == "scoped_identifier":
            return collect_scoped_path(child)
    msg = f"no scoped_identifier in: {code}"
    raise AssertionError(msg)


@skip_no_rust
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("use std::collections::HashMap;", ["std", "collections", "HashMap"]),
        ("use crate::models;", ["crate", "models"]),
        ("use super::utils;", ["super", "utils"]),
    ],
    ids=["nested-path", "crate-relative", "super-relative"],
)
def test_collect_scoped_path(code: str, expected: list[str]) -> None:
    assert _collect_from_use(code) == expected


# ── LanguageRegistration defaults ────────────────────────────────────


def test_registration_defaults() -> None:
    reg = LanguageRegistration(id="test")
    assert reg.extensions == frozenset()
    assert reg.filenames == frozenset()
    assert reg.grammar_module is None
    assert reg.grammar_entry == "language"
    assert reg.query is None
    assert reg.import_extractor is None
    assert reg.scope_types == DEFAULT_SCOPE_TYPES


def test_registration_is_frozen() -> None:
    reg = LanguageRegistration(id="test")
    with pytest.raises(AttributeError):
        reg.id = "other"  # type: ignore[misc]  # testing frozen


@pytest.mark.parametrize(
    "scope_types",
    [
        frozenset({"impl_item"}),
        frozenset(),
    ],
    ids=["custom", "empty"],
)
def test_registration_scope_types(scope_types: frozenset[str]) -> None:
    reg = LanguageRegistration(id="test", scope_types=scope_types)
    assert reg.scope_types == scope_types


def test_registration_with_all_fields() -> None:
    def dummy_extractor(node: Node) -> ImportMeta:
        return {}

    reg = LanguageRegistration(
        id="test",
        extensions=frozenset({".tst"}),
        filenames=frozenset({"Testfile"}),
        grammar_module="tree_sitter_test",
        grammar_entry="language_test",
        query="(test) @function",
        import_extractor=dummy_extractor,
        scope_types=frozenset({"test_scope"}),
    )
    assert reg.id == "test"
    assert reg.extensions == frozenset({".tst"})
    assert reg.filenames == frozenset({"Testfile"})
    assert reg.grammar_module == "tree_sitter_test"
    assert reg.grammar_entry == "language_test"
    assert reg.query == "(test) @function"
    assert reg.import_extractor is dummy_extractor
    assert reg.scope_types == frozenset({"test_scope"})


# ── DEFAULT_SCOPE_TYPES ─────────────────────────────────────────────


def test_default_scope_types_contents() -> None:
    assert "class_definition" in DEFAULT_SCOPE_TYPES
    assert "class_declaration" in DEFAULT_SCOPE_TYPES
    assert len(DEFAULT_SCOPE_TYPES) == 2


# ── ID validation ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_id",
    ["Python", "c++", "c sharp", "123lang", ""],
    ids=["uppercase", "special-char", "space", "starts-with-digit", "empty"],
)
def test_registration_rejects_invalid_id(bad_id: str) -> None:
    with pytest.raises(ValueError, match="invalid language id"):
        LanguageRegistration(id=bad_id)


def test_registration_accepts_valid_ids() -> None:
    """Lowercase, underscores, digits after first char are all fine."""
    for valid in ("python", "c", "c_sharp", "tree2"):
        reg = LanguageRegistration(id=valid)
        assert reg.id == valid
