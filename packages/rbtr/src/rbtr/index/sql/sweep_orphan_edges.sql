-- Delete edges rows whose (repo_id, snapshot_sha) has no
-- matching indexed_snapshots row.  Residue from a crashed build.
DELETE FROM edges
WHERE
  repo_id = $repo_id
  AND NOT EXISTS (
    SELECT 1 FROM indexed_snapshots AS ic
    WHERE
      ic.repo_id = edges.repo_id
      AND ic.snapshot_sha = edges.snapshot_sha
  )
