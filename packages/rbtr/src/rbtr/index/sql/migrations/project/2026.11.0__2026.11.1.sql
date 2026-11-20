-- `metadata` loses `dots`, since an import records its module as written
-- (`./b`, `..b`, `super::b`), and gains the names an import binds.  Every
-- import gets them empty, so a stored row says "none" one way; the
-- languages that record them are extracted again, by their serials.
-- DuckDB alters no column of a table an index depends on, so
-- `idx_chunks_blob` is dropped and made again.
DROP INDEX idx_chunks_blob;

ALTER TABLE chunks ALTER metadata TYPE STRUCT(
  module TEXT,
  names TEXT,
  language_hint TEXT,
  node_types TEXT[],
  bindings STRUCT(name TEXT, local TEXT)[]
) USING CASE
  WHEN kind = 'import'
    THEN struct_pack(
      module := metadata.module,
      names := metadata.names,
      language_hint := metadata.language_hint,
      node_types := NULL::TEXT[],
      bindings := []::STRUCT(name TEXT, local TEXT)[]
    )
  WHEN kind = 'fragment'
    THEN struct_pack(
      module := NULL::TEXT,
      names := NULL::TEXT,
      language_hint := NULL::TEXT,
      node_types := metadata.node_types,
      bindings := NULL::STRUCT(name TEXT, local TEXT)[]
    )
END;

CREATE INDEX idx_chunks_blob
ON chunks (blob_sha, file_language);
