"""Which chunk kinds the eval measures.

Every `ChunkKind` is a retrieval target and gets queries generated for
it, except those in `EXCLUDED_KINDS`.  A new `ChunkKind` in the indexer
is measured automatically; excluding one is a single line here.

`classify_query` shapes and `provenance` are separate axes; this file
concerns only *which target chunks* the harness generates queries for.
"""

from __future__ import annotations

from rbtr.index.models import ChunkKind

# No queries are generated for imports: an import references a symbol
# defined elsewhere, so the thing worth finding is the definition, not
# the import line.
EXCLUDED_KINDS: frozenset[ChunkKind] = frozenset({ChunkKind.IMPORT})
