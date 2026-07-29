SELECT count(*)
FROM edges
WHERE repo_id = $repo_id AND snapshot_sha = $snapshot_sha;
