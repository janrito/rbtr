SELECT count(*)
FROM file_snapshots
WHERE repo_id = $repo_id AND snapshot_sha = $snapshot_sha;
