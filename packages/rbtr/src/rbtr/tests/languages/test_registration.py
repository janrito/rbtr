"""Tests for `LanguageRegistration` and the plugin-author utilities."""

from __future__ import annotations

import functools

import pytest
from tree_sitter import Node

from rbtr.languages.registration import (
    LanguageRegistration,
    NameResolver,
    QueryExtraction,
)
from rbtr.languages.resolvers import DefaultName, DefaultScope


def test_registration_defaults() -> None:
    reg = LanguageRegistration(id="test")
    assert reg.extensions == frozenset()
    assert reg.filenames == frozenset()
    assert reg.grammar_module is None
    assert reg.grammar_entry == "language"
    assert reg.extraction is None
    # No overrides set — slots hold the built-in resolver (null-object).
    assert isinstance(reg._name_resolver, DefaultName)
    assert isinstance(reg._scope_resolver, DefaultScope)


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
def test_query_extraction_scope_types(scope_types: frozenset[str]) -> None:
    extraction = QueryExtraction(query="(x) @function", scope_types=scope_types)
    assert extraction.scope_types == scope_types


def test_query_extraction_class_scope_defaults_to_scope() -> None:
    """`class_scope_types` defaults to `scope_types` when unset."""
    extraction = QueryExtraction(query="(x) @c", scope_types=frozenset({"s"}))
    assert extraction.class_scope_types == frozenset({"s"})


def test_registration_with_all_fields() -> None:
    reg = LanguageRegistration(
        id="test",
        extensions=frozenset({".tst"}),
        filenames=frozenset({"Testfile"}),
        grammar_module="tree_sitter_test",
        grammar_entry="language_test",
        extraction=QueryExtraction(
            query="(test) @function",
            scope_types=frozenset({"test_scope"}),
        ),
    )
    assert reg.id == "test"
    assert reg.extensions == frozenset({".tst"})
    assert reg.filenames == frozenset({"Testfile"})
    assert reg.grammar_module == "tree_sitter_test"
    assert reg.grammar_entry == "language_test"
    assert isinstance(reg.extraction, QueryExtraction)
    assert reg.extraction.query == "(test) @function"
    assert reg.extraction.scope_types == frozenset({"test_scope"})


def test_registration_override_decorator() -> None:
    """The override methods work as decorators and compose over the slot."""
    reg = LanguageRegistration(id="test")

    @reg.name_extractor
    def _name(
        resolver: NameResolver, capture_name: str, node: Node, captures: dict[str, list[Node]]
    ) -> str:
        return "custom"

    assert isinstance(reg._name_resolver, functools.partial)
    assert reg._name_resolver.func is _name  # composed over the built-in
    assert isinstance(reg._scope_resolver, DefaultScope)  # untouched


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
