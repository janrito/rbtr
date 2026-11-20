"""Scope addresses and chunk ids.

Pure, dependency-light helpers in a leaf module, so the `Chunk` model can
use them without importing `chunks.py` (which imports the model) —
avoiding a circular import — and the store can compute ids over columns
of rows the same way.
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


def chunk_id(blob_sha: str, file_language: str, start_byte: int, end_byte: int) -> str:
    """A hash of where a chunk is: its blob, the language its file was read as, its byte span.

    Byte-identical files share every chunk, so a copy of an indexed file
    resolves to the row already there. `file_language` is part of it
    because the same bytes read as two languages are two extractions. A
    span of a file is one chunk, whatever its kind and name, so two
    chunks of one extraction share an id only if they share a span; bytes,
    not lines, because two chunks can share every line (`kind = 1; kind
    = 2`).
    """
    raw = f"{blob_sha}:{file_language}:{start_byte}:{end_byte}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
