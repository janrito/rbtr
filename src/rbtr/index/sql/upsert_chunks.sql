INSERT INTO chunks
SELECT
  id,
  blob_sha,
  file_path,
  kind,
  name,
  scope,
  content,
  content_tokens,
  name_tokens,
  line_start,
  line_end,
  metadata,
  NULL AS embedding
FROM _stg
ON CONFLICT (id) DO UPDATE SET
  blob_sha = excluded.blob_sha,
  file_path = excluded.file_path,
  kind = excluded.kind,
  name = excluded.name,
  scope = excluded.scope,
  content = excluded.content,
  content_tokens = excluded.content_tokens,
  name_tokens = excluded.name_tokens,
  line_start = excluded.line_start,
  line_end = excluded.line_end,
  metadata = excluded.metadata
