-- Upsert: insert a new completion row, or refresh indexed_at on
-- re-index. Idempotent; safe to call on retries. `now()` is used
-- instead of `current_timestamp` because DuckDB parses the latter
-- as an unqualified column reference in the SET clause.
INSERT INTO indexed_snapshots (repo_id, snapshot_sha)
VALUES ($repo_id, $snapshot_sha)
ON CONFLICT (repo_id, snapshot_sha)
DO UPDATE SET indexed_at = now();
