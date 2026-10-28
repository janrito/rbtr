-- `NULL` clears it: the checkout's path resolves again.
UPDATE repos
SET missing_since = $missing_since
WHERE id = $repo_id
