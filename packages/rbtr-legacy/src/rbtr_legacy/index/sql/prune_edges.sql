DELETE FROM edges
WHERE
  commit_sha NOT IN (SELECT DISTINCT commit_sha FROM file_snapshots)
