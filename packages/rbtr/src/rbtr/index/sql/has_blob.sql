SELECT 1
FROM chunks
WHERE blob_sha = ?
LIMIT 1 -- noqa: AM09
