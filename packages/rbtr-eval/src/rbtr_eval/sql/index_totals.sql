SELECT
  (
    SELECT COUNT(DISTINCT c.id)
    FROM indexed_snapshots AS s
    INNER JOIN file_snapshots AS fs
      ON
        s.repo_id = fs.repo_id
        AND s.snapshot_sha = fs.snapshot_sha
    INNER JOIN chunks AS c
      ON
        fs.blob_sha = c.blob_sha
        AND fs.detected_language = c.file_language
  ) AS total_chunks,
  (
    SELECT COUNT(*)
    FROM indexed_snapshots AS s
    INNER JOIN file_snapshots AS fs
      ON
        s.repo_id = fs.repo_id
        AND s.snapshot_sha = fs.snapshot_sha
    INNER JOIN chunks AS c
      ON
        fs.blob_sha = c.blob_sha
        AND fs.detected_language = c.file_language
  ) AS total_locations,
  (
    SELECT COUNT(*)
    FROM indexed_snapshots AS s
    INNER JOIN edges AS e
      ON
        s.repo_id = e.repo_id
        AND s.snapshot_sha = e.snapshot_sha
  ) AS total_edges
