WITH repo_chunks AS (
  SELECT
    fs.repo_id,
    COUNT(DISTINCT c.id) AS chunks,
    COUNT(DISTINCT c.id) FILTER (c.embedding IS NOT NULL) AS embedded,
    COUNT(DISTINCT c.id) FILTER (c.embedding_truncated) AS truncated
  FROM file_snapshots AS fs
  INNER JOIN chunks AS c
    ON
      fs.blob_sha = c.blob_sha
      AND fs.file_path = c.file_path
  GROUP BY fs.repo_id
)

SELECT
  r.path,
  COALESCE(rc.chunks, 0) AS chunks,
  COALESCE(rc.embedded, 0) AS embedded,
  COALESCE(rc.truncated, 0) AS truncated
FROM repos AS r
LEFT JOIN repo_chunks AS rc ON r.id = rc.repo_id
ORDER BY r.path
