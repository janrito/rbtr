INSERT INTO chunks
SELECT
  id,
  blob_sha,
  kind,
  name,
  scope,
  language,
  file_language,
  extraction_serial,
  content,
  content_tokens,
  name_tokens,
  line_start,
  line_end,
  metadata,
  NULL AS embedding,
  FALSE AS embedding_truncated
FROM _stg
ON CONFLICT (id) DO UPDATE SET
  blob_sha = excluded.blob_sha,
  kind = excluded.kind,
  name = excluded.name,
  scope = excluded.scope,
  language = excluded.language,
  file_language = excluded.file_language,
  extraction_serial = excluded.extraction_serial,
  content = excluded.content,
  content_tokens = excluded.content_tokens,
  name_tokens = excluded.name_tokens,
  line_start = excluded.line_start,
  line_end = excluded.line_end,
  metadata = excluded.metadata,
  -- The model reads a chunk's name, part of its id, and its content: the
  -- embedding still holds while the content does.
  embedding
  = CASE WHEN chunks.content = excluded.content THEN chunks.embedding END,
  embedding_truncated = (
    chunks.content = excluded.content AND chunks.embedding_truncated
  )
