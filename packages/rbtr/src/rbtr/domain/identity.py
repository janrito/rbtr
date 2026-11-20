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


def chunk_id(
    blob_sha: str, file_language: str, kind: str, name: str, line_start: int, line_end: int
) -> str:
    """A hash of what a chunk is: its blob, the language its file was read as, its role, its span.

    Byte-identical files share every chunk, so a copy of an indexed file
    resolves to the row already there. `file_language` is part of it
    because the same bytes read as two languages are two extractions;
    `kind` and `line_end` because a chunk is a span with a role, and an
    rst `doc_section` and a `:mod:` reference can share a name and a
    starting line while spanning different lines.
    """
    raw = f"{blob_sha}:{file_language}:{kind}:{name}:{line_start}:{line_end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
