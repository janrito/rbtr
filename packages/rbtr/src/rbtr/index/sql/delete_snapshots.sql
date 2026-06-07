DELETE FROM file_snapshots
WHERE repo_id = $repo_id AND commit_sha = $commit_sha
