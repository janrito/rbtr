SELECT 1 FROM indexed_snapshots
WHERE repo_id = $repo_id AND snapshot_sha = $snapshot_sha
LIMIT 1 -- noqa: AM09
