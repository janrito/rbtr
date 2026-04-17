-- Delete the indexed_commits completion row for this commit.
DELETE FROM indexed_commits WHERE repo_id = ? AND commit_sha = ?;
