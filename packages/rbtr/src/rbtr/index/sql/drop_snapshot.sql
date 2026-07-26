-- Delete the indexed_snapshots completion row for this snapshot.
DELETE FROM indexed_snapshots
WHERE repo_id = $repo_id AND snapshot_sha = $snapshot_sha;
