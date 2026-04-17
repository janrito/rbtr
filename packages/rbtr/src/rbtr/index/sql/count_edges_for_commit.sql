SELECT count(*)
FROM edges
WHERE repo_id = ? AND commit_sha = ?;
