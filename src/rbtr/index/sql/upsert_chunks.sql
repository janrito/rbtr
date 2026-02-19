INSERT INTO chunks
SELECT
  id,
  blob_sha,
  file_path,
  kind,
  name,
  scope,
  content,
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
  line_start = excluded.line_start,
  line_end = excluded.line_end,
  metadata = excluded.metadata
