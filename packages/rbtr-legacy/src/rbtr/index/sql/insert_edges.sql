INSERT INTO edges
SELECT
  source_id,
  target_id,
  kind,
  commit_sha
FROM _stg
