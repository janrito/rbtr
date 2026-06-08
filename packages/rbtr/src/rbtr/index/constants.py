"""Shared constants for the index: schema and embedding format versions.

These live in their own module to avoid circular imports:
`store.py` and `writer.py` need them, and `store` already imports
from `frames`.
"""

from __future__ import annotations

# Bump this constant any time the schema changes in a way that
# can't be migrated (column add/remove/retype).  On open, if the
# stored version in `meta.schema_version` doesn't match the code's
# version, the DB file is deleted and rebuilt from scratch.  Uses
# calver (YYYY.M.NUM) matching the project version format.
SCHEMA_VERSION = "2026.5.5"

# Bump this when the embedding text format (i.e. `embedding_text()`)
# changes.  Stored alongside `config.embedding_model` in the DB's
# `meta` table; a mismatch in either value clears all embeddings so
# `_embed_missing()` re-computes them on the next index build.
EMBEDDING_FORMAT_VERSION = 2
