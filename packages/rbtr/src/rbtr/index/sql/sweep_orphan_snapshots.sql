-- Delete file_snapshots rows whose (repo_id, snapshot_sha) has no
-- matching indexed_snapshots row. These are residue from a build
-- that crashed before it could mark the commit complete.
DELETE FROM file_snapshots
WHERE
  repo_id = $repo_id
  AND NOT EXISTS (
    SELECT 1 FROM indexed_snapshots AS ic
    WHERE
      ic.repo_id = file_snapshots.repo_id
      AND ic.snapshot_sha = file_snapshots.snapshot_sha
  )
