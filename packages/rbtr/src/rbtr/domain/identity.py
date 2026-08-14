"""Scope-address composition.

A pure, dependency-light helper in a leaf module, so the `Chunk` model
can use it in a validator without importing `chunks.py` (which imports
the model) — avoiding a circular import. Chunk identity itself is the
`Chunk.id` property: it hashes fields of the model, so it has no
caller that does not already hold a chunk.
"""

from __future__ import annotations

from collections.abc import Iterable

# Separator joining enclosing-scope names into an address path.
# Shared so code extraction and the doc-heading chunkers agree.
SCOPE_SEPARATOR = "::"


def compose_scope(names: Iterable[str]) -> str:
    """Join enclosing-scope *names* (outermost-first) into an address.

    The single place a scope address is formed, used by the `Chunk`
    scope validator (for code extraction's ancestor walk and the
    markdown/rst heading chains alike). Empty names are dropped, so an
    anonymous or unnamed scope contributes no segment.
    """
    return SCOPE_SEPARATOR.join(n for n in names if n)
