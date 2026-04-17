SELECT commit_sha, indexed_at
FROM indexed_commits
WHERE repo_id = ?
ORDER BY indexed_at DESC;
