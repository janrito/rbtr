INSERT INTO edges
SELECT
  repo_id,
  source_id,
  target_id,
  kind,
  snapshot_sha
FROM _stg
ON CONFLICT DO NOTHING
