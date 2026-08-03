WITH scoped_chunks AS (
  SELECT DISTINCT
    c.id,
    c.embedding_truncated,
    c.embedding IS NOT NULL AS has_embedding
  FROM indexed_snapshots AS s
  INNER JOIN file_snapshots AS fs
    ON
      s.repo_id = fs.repo_id
      AND s.snapshot_sha = fs.snapshot_sha
  INNER JOIN chunks AS c
    ON
      fs.blob_sha = c.blob_sha
      AND fs.detected_language = c.file_language
)

SELECT
  COUNT(*) AS total_chunks,
  COUNT(*) FILTER (has_embedding) AS with_embedding,
  COUNT(*) FILTER (embedding_truncated) AS truncated
FROM scoped_chunks
