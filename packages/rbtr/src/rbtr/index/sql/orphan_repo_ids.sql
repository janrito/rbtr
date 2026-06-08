-- Repo IDs that have file_snapshots not backed by an
-- indexed_commits row — residue from a crashed build.
SELECT DISTINCT repo_id
FROM file_snapshots
WHERE
  NOT EXISTS (
    SELECT 1
    FROM indexed_commits AS ic
    WHERE
      ic.repo_id = file_snapshots.repo_id
      AND ic.commit_sha = file_snapshots.commit_sha
  )
