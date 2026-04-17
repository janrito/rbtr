-- Upsert: insert a new completion row, or refresh indexed_at on
-- re-index. Idempotent; safe to call on retries. `now()` is used
-- instead of `current_timestamp` because DuckDB parses the latter
-- as an unqualified column reference in the SET clause.
INSERT INTO indexed_commits (repo_id, commit_sha)
VALUES (?, ?)
ON CONFLICT (repo_id, commit_sha)
DO UPDATE SET indexed_at = now();
