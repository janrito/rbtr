SELECT 1
FROM chunks
WHERE repo_id = ? AND blob_sha = ? AND strip_docstrings = ?
LIMIT 1 -- noqa: AM09
