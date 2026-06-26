"""Shared fixtures and grammar helpers for language tests.

Exposes `language_manager` as a session-scoped fixture over
the production `LanguageManager` singleton.  Tests take it as
a parameter wherever they need grammar / query / extraction
lookup.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk
from rbtr.index.treesitter import build_import_from_captures, extract_symbols
from rbtr.languages import LanguageManager, get_manager


@pytest.fixture(scope="session")
def language_manager() -> LanguageManager:
    """The production `LanguageManager` singleton."""
    return get_manager()


def extract_chunks(
    lang: str,
    source: str,
    file_path: str = "",
    *,
    no_leading_attachment: bool = False,
) -> list[Chunk]:
    """Run the real extraction pipeline for *lang* on *source*.

    Routes through the same paths as the orchestrator:
    chunker-based plugins get their chunker called with the
    grammar; query-based plugins go through `extract_symbols`.

    Calls `get_manager()` inline because `pytest.param(...,
    marks=...)` evaluates markers at collection time, and cases
    invoking this from inside `@parametrize_with_cases` cannot
    consume a fixture.

    By default uses the plugin's registered
    `doc_comment_node_types` so callers see the same
    leading-comment attachment as production.
    `no_leading_attachment` forces an empty
    `doc_comment_node_types` to probe engine fall-back
    behaviour (interior `@_docstring` capture still fires).
    """
    manager = get_manager()
    reg = manager.get_registration(lang)
    assert reg is not None, f"no registration for {lang}"

    grammar = manager.load_grammar(lang)
    assert grammar is not None, f"grammar for {lang} not installed"

    ext = next(iter(reg.extensions), ".txt").lstrip(".")
    path = file_path or f"test.{ext}"

    # Chunker path (prose/config plugins).
    if reg.chunker is not None:
        return list(reg.chunker(path, "sha1", source, grammar))

    # Query path (code plugins).
    assert reg.query is not None, f"no query for {lang}"
    doc_types = frozenset() if no_leading_attachment else reg.doc_comment_node_types
    return list(
        extract_symbols(
            path,
            "sha1",
            source.encode(),
            grammar,
            reg.query,
            import_extractor=(
                reg.import_extractor
                if reg.import_extractor is not None
                else build_import_from_captures
            ),
            scope_types=reg.scope_types,
            class_scope_types=reg.class_scope_types,
            doc_comment_node_types=doc_types,
        )
    )


def skip_unless_grammar(lang: str) -> pytest.MarkDecorator:
    """Return a `skipif` marker when the grammar for *lang* is missing.

    Called at parametrize collection time, so it cannot depend on
    a fixture.  `get_manager()` is idempotent.
    """
    return pytest.mark.skipif(
        get_manager().load_grammar(lang) is None,
        reason=f"tree-sitter-{lang} not installed",
    )
