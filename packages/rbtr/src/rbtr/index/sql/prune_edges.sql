DELETE FROM edges
WHERE repo_id = ?
  AND commit_sha NOT IN (
    SELECT DISTINCT commit_sha
    FROM file_snapshots
    WHERE repo_id = ?
  )
