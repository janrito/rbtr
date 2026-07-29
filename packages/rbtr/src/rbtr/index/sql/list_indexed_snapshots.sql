SELECT
  snapshot_sha,
  indexed_at
FROM indexed_snapshots
WHERE repo_id = $repo_id
ORDER BY indexed_at DESC;
