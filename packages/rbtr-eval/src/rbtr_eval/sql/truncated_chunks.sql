SELECT DISTINCT
  r.path,
  c.file_path,
  c.scope,
  c.name,
  c.line_start
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
INNER JOIN repos AS r ON fs.repo_id = r.id
WHERE c.embedding_truncated
