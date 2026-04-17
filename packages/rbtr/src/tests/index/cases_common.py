"""Shared dataclass definitions for ``case_*`` files.

Holds only class definitions (no data).  Families that need the
same shape import from here; data always comes from fixtures
declared by each case function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rbtr.index.models import ChunkKind, ImportMeta


@dataclass(frozen=True)
class ChunkSpec:
    """The fields of ``Chunk`` that case functions care about.

    Every family uses this shape; families that don't care about
    ``content`` or ``metadata`` leave them at default.  Required
    pydantic fields (``blob_sha``, ``line_start``, ``line_end``)
    are filled in by each test fixture when constructing real
    ``Chunk`` instances.
    """

    id: str
    kind: ChunkKind = ChunkKind.FUNCTION
    name: str = "fn"
    file_path: str = "src/lib.py"
    content: str = ""
    metadata: ImportMeta = field(default_factory=dict)
