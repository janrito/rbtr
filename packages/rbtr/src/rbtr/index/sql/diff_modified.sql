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
INNER JOIN file_snapshots AS head_fs
  ON
    c.repo_id = head_fs.repo_id
    AND c.blob_sha = head_fs.blob_sha
    AND c.file_path = head_fs.file_path
INNER JOIN file_snapshots AS base_fs
  ON
    head_fs.repo_id = base_fs.repo_id
    AND head_fs.file_path = base_fs.file_path
WHERE
  head_fs.repo_id = ?
  AND head_fs.commit_sha = ?
  AND base_fs.commit_sha = ?
  AND head_fs.blob_sha <> base_fs.blob_sha
ORDER BY c.file_path, c.line_start
