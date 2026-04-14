SELECT
  c.id,
  c.blob_sha,
  c.file_path,
  c.kind,
  c.name,
  c.scope,
  c.content,
  c.line_start,
  c.line_end,
  c.metadata,
  c.embedding IS NOT NULL AS has_embedding
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
WHERE
  fs.commit_sha = ?
  AND fs.file_path NOT IN (
    SELECT file_snapshots.file_path FROM file_snapshots
    WHERE file_snapshots.commit_sha = ?
  )
ORDER BY c.file_path, c.line_start
