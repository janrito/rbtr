-- sqlfluff:templater:placeholder:pattern:'%main%'
SELECT
  c.id,
  c.repo_id,
  c.blob_sha,
  c.file_path,
  c.kind,
  c.name,
  c.scope,
  c.language,
  c.content,
  c.line_start,
  c.line_end,
  c.metadata,
  c.embedding IS NOT NULL AS has_embedding
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.repo_id = fs.repo_id
    AND c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
INNER JOIN _repo_refs AS rr
  ON fs.repo_id = rr.repo_id AND fs.commit_sha = rr.commit_sha
WHERE c.name ILIKE $pattern
ORDER BY c.file_path, c.line_start
