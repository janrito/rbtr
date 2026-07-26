DELETE FROM edges
WHERE
  repo_id = $repo_id
  AND snapshot_sha NOT IN (
    SELECT DISTINCT snapshot_sha
    FROM file_snapshots
    WHERE repo_id = $repo_id
  )
