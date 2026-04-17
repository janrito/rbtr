SELECT 1 FROM indexed_commits
WHERE repo_id = ? AND commit_sha = ?
LIMIT 1;
