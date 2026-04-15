INSERT INTO edges
SELECT
  repo_id,
  source_id,
  target_id,
  kind,
  commit_sha
FROM _stg
