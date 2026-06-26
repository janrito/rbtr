SELECT
  c.id,
  fs.repo_id,
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
  c.embedding IS NOT NULL AS has_embedding,
  MAX(LIST_COSINE_SIMILARITY(c.embedding, q.vec)) AS score
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
INNER JOIN _repo_refs AS rr
  ON fs.repo_id = rr.repo_id AND fs.commit_sha = rr.commit_sha
CROSS JOIN _qvecs AS q
WHERE c.embedding IS NOT NULL
GROUP BY ALL
ORDER BY score DESC
LIMIT $top_k
