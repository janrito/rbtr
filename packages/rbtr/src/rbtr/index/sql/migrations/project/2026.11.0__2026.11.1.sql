-- An import records a relative module as written (`./b`, `../b`), so
-- `metadata` loses `dots`.  The languages that set it are extracted
-- again, by their serials.  DuckDB alters no column of a table an index
-- depends on, so `idx_chunks_blob` is dropped and made again.
DROP INDEX idx_chunks_blob;

ALTER TABLE chunks ALTER metadata TYPE STRUCT(
  module TEXT, names TEXT, language_hint TEXT, node_types TEXT[]
) USING CASE
  WHEN metadata IS NOT NULL
    THEN struct_pack(
      module := metadata.module,
      names := metadata.names,
      language_hint := metadata.language_hint,
      node_types := metadata.node_types
    )
END;

CREATE INDEX idx_chunks_blob
ON chunks (blob_sha, file_language);
