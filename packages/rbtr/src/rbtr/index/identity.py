"""Chunk identity and scope-address primitives.

Pure, dependency-light helpers shared by the `Chunk` model, the
extractor, and the language plugins. They live in a leaf module so the
model can use them in its validators without importing `chunks.py`
(which imports the model) — avoiding a circular import.
"""

from __future__ import annotations

import hashlib
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


def make_chunk_id(file_path: str, blob_sha: str, name: str, line_start: int) -> str:
    """Deterministic chunk ID from file path, blob SHA, symbol name, and line."""
    raw = f"{file_path}:{blob_sha}:{name}:{line_start}"
    return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()
