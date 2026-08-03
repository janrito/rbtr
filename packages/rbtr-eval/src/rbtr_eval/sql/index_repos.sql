WITH repo_chunks AS (
  SELECT
    fs.repo_id,
    COUNT(DISTINCT c.id) AS chunks,
    COUNT(*) AS locations
  FROM indexed_snapshots AS s
  INNER JOIN file_snapshots AS fs
    ON
      s.repo_id = fs.repo_id
      AND s.snapshot_sha = fs.snapshot_sha
  INNER JOIN chunks AS c
    ON
      fs.blob_sha = c.blob_sha
      AND fs.detected_language = c.file_language
  GROUP BY fs.repo_id
),

repo_edges AS (
  SELECT
    e.repo_id,
    COUNT(*) AS edges
  FROM indexed_snapshots AS s
  INNER JOIN edges AS e
    ON
      s.repo_id = e.repo_id
      AND s.snapshot_sha = e.snapshot_sha
  GROUP BY e.repo_id
)

SELECT
  r.path,
  COALESCE(rc.chunks, 0) AS chunks,
  COALESCE(rc.locations, 0) AS locations,
  COALESCE(re.edges, 0) AS edges
FROM repos AS r
LEFT JOIN repo_chunks AS rc ON r.id = rc.repo_id
LEFT JOIN repo_edges AS re ON r.id = re.repo_id
ORDER BY r.path
