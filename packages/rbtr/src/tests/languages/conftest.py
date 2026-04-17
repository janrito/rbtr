"""Shared fixtures and grammar helpers for language tests.

Exposes ``language_manager`` as a session-scoped fixture over
the production ``LanguageManager`` singleton.  Tests take it as
a parameter wherever they need grammar / query / extraction
lookup.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk
from rbtr.index.treesitter import extract_symbols
from rbtr.languages import LanguageManager, get_manager


@pytest.fixture(scope="session")
def language_manager() -> LanguageManager:
    """The production ``LanguageManager`` singleton."""
    return get_manager()


def extract_chunks(
    lang: str,
    source: str,
    file_path: str = "",
) -> list[Chunk]:
    """Run the real extraction pipeline for *lang* on *source*.

    Invokes the system under test — same class of helper as
    ``_run`` (subprocess invocation) and ``_lf`` (``list_files``
    wrapper) elsewhere: not setup, but a one-liner shorthand over
    caller-supplied arguments.  Calls ``get_manager()`` inline
    because ``pytest.param(..., marks=...)`` evaluates markers at
    collection time, and cases invoking this from inside
    ``@parametrize_with_cases`` cannot consume a fixture.
    """
    manager = get_manager()
    grammar = manager.load_grammar(lang)
    assert grammar is not None, f"grammar for {lang} not installed"
    reg = manager.get_registration(lang)
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
    """Return a ``skipif`` marker when the grammar for *lang* is missing.

    Called at parametrize collection time, so it cannot depend on
    a fixture.  ``get_manager()`` is idempotent.
    """
    return pytest.mark.skipif(
        get_manager().load_grammar(lang) is None,
        reason=f"tree-sitter-{lang} not installed",
    )
