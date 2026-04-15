SELECT 1
FROM chunks
WHERE repo_id = ? AND blob_sha = ?
LIMIT 1 -- noqa: AM09
