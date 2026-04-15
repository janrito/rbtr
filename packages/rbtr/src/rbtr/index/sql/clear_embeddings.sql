UPDATE chunks
SET embedding = NULL
WHERE repo_id = ? AND embedding IS NOT NULL
