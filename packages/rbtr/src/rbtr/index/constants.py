"""Shared constants for the index: schema and embedding format versions.

These live in their own module to avoid circular imports:
`index.py` and `writer.py` need them, and `index` already imports
from `results`.
"""

from __future__ import annotations

# The schema version this code reads and writes.  Every index file
# records in `meta.schema_version` the version that last migrated it.
# Bump this with every schema change, and add a migration for each
# kind of file the change touches (`index/migrations.py`).  Uses
# calver (YYYY.M.NUM) matching the project version format.
SCHEMA_VERSION = "2026.11.0"

# Bump this when the embedding text format (i.e. `embedding_text()`)
# changes.  Stored alongside `config.embedding_model` in the DB's
# `meta` table; a mismatch in either value clears all embeddings so
# `_embed_missing()` re-computes them on the next index build.
EMBEDDING_FORMAT_VERSION = 2
