"""Behaviour tests for extract query generation.

Docstring detection uses tree-sitter's `extract_doc_spans`
on the chunk content. Tested through the observable output:
which provenances are produced for each symbol.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr_eval.extract import queries_for_symbol


@parametrize_with_cases(
    "content, language, name, expected_provenances",
    cases=".cases_extract",
    has_tag="yields_queries",
)
def test_generates_expected_provenances(
    content: str, language: str, name: str, expected_provenances: set[str]
) -> None:
    """Symbol produces the expected set of provenances."""
    queries = queries_for_symbol(
        slug="test",
        file_path="test.py",
        scope="",
        name=name,
        symbol_kind="function",
        line_start=1,
        language=language,
        content=content,
    )
    actual_provenances = {q["provenance"] for q in queries}
    assert actual_provenances == expected_provenances
