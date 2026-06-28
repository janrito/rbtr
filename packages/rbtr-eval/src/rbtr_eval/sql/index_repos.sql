WITH repo_chunks AS (
  SELECT
    fs.repo_id,
    COUNT(DISTINCT c.id) AS chunks
  FROM file_snapshots AS fs
  INNER JOIN chunks AS c
    ON
      fs.blob_sha = c.blob_sha
      AND fs.file_path = c.file_path
  GROUP BY fs.repo_id
),

repo_edges AS (
  SELECT
    repo_id,
    COUNT(*) AS edges
  FROM edges
  GROUP BY repo_id
)

SELECT
  r.path,
  COALESCE(rc.chunks, 0) AS chunks,
  COALESCE(re.edges, 0) AS edges
FROM repos AS r
LEFT JOIN repo_chunks AS rc ON r.id = rc.repo_id
LEFT JOIN repo_edges AS re ON r.id = re.repo_id
ORDER BY r.path
