-- Content-addressed store: chunks are shared across a project's checkouts
-- and paths, so a blob's chunks are deleted for all of them (chunking is
-- deterministic per extraction serial, so every checkout sees the same
-- chunks for a blob).
--
-- Scoped to one `file_language`, because the same bytes extracted as a
-- second language are different chunks: deleting by blob alone would
-- take the first language's chunks with it and leave that file with
-- nothing to find.
--
-- Ids in `$keep` stay: a re-extraction producing a chunk again upserts it,
-- keeping its embedding while its content is unchanged.
DELETE FROM chunks
WHERE
  blob_sha IN (SELECT unnest($blob_shas::TEXT[]))
  AND file_language = $file_language
  AND id NOT IN (SELECT unnest($keep::TEXT[]))
