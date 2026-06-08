SELECT count(*)
FROM edges
WHERE repo_id = $repo_id AND commit_sha = $commit_sha;
