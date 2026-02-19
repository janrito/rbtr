"""Language detection and grammar loading — public API.

Thin delegation to :mod:`rbtr.plugins.manager`.  All language
behaviour is provided by plugins; this module re-exports the
convenience functions that the rest of the codebase uses.

Language IDs are plain strings matching the ``id`` field on
``LanguageRegistration``.  The canonical set is defined by the
plugins registered in ``manager.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbtr.plugins.manager import get_manager

if TYPE_CHECKING:
    from tree_sitter import Language


def detect_language(file_path: str) -> str | None:
    """Detect the language of a file from its extension or filename.

    Returns a language ID string, or ``None`` for unrecognised files.
    """
    return get_manager().detect_language(file_path)


def load_grammar(lang: str) -> Language | None:
    """Load the tree-sitter grammar for *lang*.

    Returns the ``Language`` object, or ``None`` if the grammar
    is not configured or not installed.  Results are memoised.
    """
    return get_manager().load_grammar(lang)


def get_language(file_path: str) -> tuple[str, Language] | None:
    """Detect language and load its grammar in one step.

    Returns ``(language_id, grammar)`` or ``None``.
    """
    return get_manager().get_language(file_path)


def missing_grammar(lang: str) -> bool:
    """Check whether a language was detected but has no grammar installed."""
    return get_manager().missing_grammar(lang)
