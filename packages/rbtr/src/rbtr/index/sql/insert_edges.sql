INSERT INTO edges
SELECT
  repo_id,
  source_id,
  target_id,
  kind,
  snapshot_sha,
  source_path,
  target_path
FROM _stg
ON CONFLICT DO NOTHING
