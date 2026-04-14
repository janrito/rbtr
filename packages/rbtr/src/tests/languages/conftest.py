"""Shared fixtures for plugin extraction tests."""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk
from rbtr.index.treesitter import extract_symbols
from rbtr.languages import get_manager

_manager = get_manager()


def extract_chunks(
    lang: str,
    source: str,
    file_path: str = "",
) -> list[Chunk]:
    """Extract chunks for *lang* from *source* via the plugin system.

    Uses `get_manager()` to load grammar, query, extractor, and
    scope types — the same path the production code takes.
    """
    grammar = _manager.load_grammar(lang)
    assert grammar is not None, f"grammar for {lang} not installed"
    reg = _manager.get_registration(lang)
    assert reg is not None
    assert reg.query is not None, f"no query for {lang}"

    ext = next(iter(reg.extensions), ".txt").lstrip(".")
    path = file_path or f"test.{ext}"
    return extract_symbols(
        path,
        "sha1",
        source.encode(),
        grammar,
        reg.query,
        import_extractor=reg.import_extractor,
        scope_types=reg.scope_types,
    )


def skip_unless_grammar(lang: str) -> pytest.MarkDecorator:
    """Return a `skipif` marker when the grammar for *lang* is missing."""
    return pytest.mark.skipif(
        _manager.load_grammar(lang) is None,
        reason=f"tree-sitter-{lang} not installed",
    )
