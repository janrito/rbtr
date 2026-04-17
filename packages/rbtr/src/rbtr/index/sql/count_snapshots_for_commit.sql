SELECT count(*)
FROM file_snapshots
WHERE repo_id = ? AND commit_sha = ?;
