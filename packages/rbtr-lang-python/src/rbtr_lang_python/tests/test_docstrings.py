"""Python docstring extraction.

rbtr folds a symbol's docstring into its chunk content. Python is an
interior-doc language: the `@_docstring` query capture holds the docstring,
so `line_start` stays at the `def`. Data lives in `cases_docstrings.py`,
sliced by tag.
"""

from __future__ import annotations

from pytest_cases import parametrize_with_cases

from rbtr.testing import Indexed
from rbtr.testing.reads import read


@parametrize_with_cases("files, name, snippet", cases=".cases_docstrings", has_tag="documented")
def test_documented_chunk_includes_doc_text(
    indexed: Indexed, files: dict[str, str], name: str, snippet: str
) -> None:
    """By default the chunk content carries the symbol's docs."""
    with indexed(files) as build:
        [chunk] = read(build, name)
    assert snippet in chunk.content, (
        f"expected {snippet!r} in {name} content; got:\n{chunk.content!r}"
    )


@parametrize_with_cases("files, name, snippet", cases=".cases_docstrings", has_tag="undocumented")
def test_no_phantom_documentation(
    indexed: Indexed, files: dict[str, str], name: str, snippet: str
) -> None:
    """Symbols without documentation do not gain any in content."""
    with indexed(files) as build:
        [chunk] = read(build, name)
    assert snippet not in chunk.content, (
        f"unexpected {snippet!r} in {name} content; got:\n{chunk.content!r}"
    )
