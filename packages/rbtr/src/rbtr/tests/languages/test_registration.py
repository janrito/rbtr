"""Tests for `LanguageRegistration` and the plugin-author utilities."""

from __future__ import annotations

import pytest
from tree_sitter import Node

from rbtr.index.models import ImportMeta
from rbtr.languages.registration import (
    ImportResolver,
    LanguageRegistration,
    NameResolver,
    parse_path_relative,
)

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


# ── LanguageRegistration defaults ────────────────────────────────────


def test_registration_defaults() -> None:
    reg = LanguageRegistration(id="test")
    assert reg.extensions == frozenset()
    assert reg.filenames == frozenset()
    assert reg.grammar_module is None
    assert reg.grammar_entry == "language"
    assert reg.query is None
    assert reg.scope_types == frozenset()
    # No overrides set — slots are None; the engine substitutes defaults.
    assert reg._name_extractor is None
    assert reg._scope_extractor is None
    assert reg._import_extractor is None
    assert reg._chunker is None


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
    def dummy_extractor(
        resolver: ImportResolver, node: Node, captures: dict[str, list[Node]]
    ) -> ImportMeta:
        return ImportMeta()

    reg = LanguageRegistration(
        id="test",
        extensions=frozenset({".tst"}),
        filenames=frozenset({"Testfile"}),
        grammar_module="tree_sitter_test",
        grammar_entry="language_test",
        query="(test) @function",
        scope_types=frozenset({"test_scope"}),
    )
    reg.import_extractor(dummy_extractor)  # attach via the method
    assert reg.id == "test"
    assert reg.extensions == frozenset({".tst"})
    assert reg.filenames == frozenset({"Testfile"})
    assert reg.grammar_module == "tree_sitter_test"
    assert reg.grammar_entry == "language_test"
    assert reg.query == "(test) @function"
    assert reg._import_extractor is dummy_extractor
    assert reg.scope_types == frozenset({"test_scope"})


def test_registration_override_decorator() -> None:
    """The override methods work as decorators and set the matching slot."""
    reg = LanguageRegistration(id="test")

    @reg.name_extractor
    def _name(
        resolver: NameResolver, capture_name: str, node: Node, captures: dict[str, list[Node]]
    ) -> str:
        return "custom"

    assert reg._name_extractor is _name
    assert reg._scope_extractor is None  # untouched


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
