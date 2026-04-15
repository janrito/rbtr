SELECT
  c.id,
  c.blob_sha,
  c.file_path,
  c.kind,
  c.name,
  c.scope,
  c.content,
  c.content_tokens,
  c.name_tokens,
  c.line_start,
  c.line_end,
  c.metadata,
  c.embedding IS NOT NULL AS has_embedding,
  list_cosine_similarity(c.embedding, ?::FLOAT []) AS score
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.repo_id = fs.repo_id
    AND c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
WHERE
  c.repo_id = ?
  AND fs.commit_sha = ?
  AND c.embedding IS NOT NULL
ORDER BY score DESC
LIMIT ?
