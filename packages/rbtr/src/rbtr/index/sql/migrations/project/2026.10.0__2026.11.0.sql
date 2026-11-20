-- `chunks.metadata` holds a chunk's role payload as a struct, not JSON
-- text: an import's metadata, and NULL on every other chunk.  Rows
-- written before held every chunk's import metadata as JSON, empty for
-- all but imports.  DuckDB alters no column of a table an index depends
-- on, so `idx_chunks_blob` is dropped and made again.
DROP INDEX idx_chunks_blob;

ALTER TABLE chunks ALTER metadata DROP DEFAULT;

ALTER TABLE chunks ALTER metadata DROP NOT NULL;

ALTER TABLE chunks ALTER metadata TYPE STRUCT(
  module TEXT, names TEXT, dots TEXT, language_hint TEXT, node_types TEXT[]
) USING CASE
  WHEN kind = 'import'
    THEN STRUCT_PACK(
      module := COALESCE(JSON_EXTRACT_STRING(metadata, '$.module'), ''),
      names := COALESCE(JSON_EXTRACT_STRING(metadata, '$.names'), ''),
      dots := COALESCE(JSON_EXTRACT_STRING(metadata, '$.dots'), ''),
      language_hint
      := COALESCE(JSON_EXTRACT_STRING(metadata, '$.language_hint'), ''),
      node_types := NULL::TEXT[]
    )
END;

CREATE INDEX idx_chunks_blob
ON chunks (blob_sha, file_language);
