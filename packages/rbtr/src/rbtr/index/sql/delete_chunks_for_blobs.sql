-- Content-addressed store: chunks are shared across repos and paths, so
-- a blob's chunks are deleted globally (chunking is deterministic per
-- extraction serial, so every repo sees the same chunks for a blob).
--
-- Scoped to one `file_language`, because the same bytes extracted as a
-- second language are different chunks: deleting by blob alone would
-- take the first language's chunks with it and leave that file with
-- nothing to find.
DELETE FROM chunks
WHERE
  blob_sha IN (SELECT unnest($blob_shas::TEXT []))
  AND file_language = $file_language
