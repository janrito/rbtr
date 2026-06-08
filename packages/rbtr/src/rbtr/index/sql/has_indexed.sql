SELECT 1 FROM indexed_commits
WHERE repo_id = $repo_id AND commit_sha = $commit_sha
LIMIT 1 -- noqa: AM09
