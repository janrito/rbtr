-- Delete file_snapshots rows whose (repo_id, commit_sha) has no
-- matching indexed_commits row. These are residue from a build
-- that crashed before it could mark the commit complete.
DELETE FROM file_snapshots
WHERE repo_id = ?
  AND NOT EXISTS (
    SELECT 1 FROM indexed_commits ic
    WHERE ic.repo_id = file_snapshots.repo_id
      AND ic.commit_sha = file_snapshots.commit_sha
  )
