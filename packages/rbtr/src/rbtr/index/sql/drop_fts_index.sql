-- Drop the FTS index, which removes the `fts_main_chunks` schema.
-- The caller guards this with `has_fts_index.sql`: `drop_fts_index`
-- errors when the index is absent.
PRAGMA drop_fts_index('chunks')  -- noqa: PRS
