SELECT
  r.path,
  c.file_path,
  c.scope,
  c.name,
  c.line_start
FROM chunks AS c
INNER JOIN repos AS r ON c.repo_id = r.id
WHERE c.embedding_truncated
