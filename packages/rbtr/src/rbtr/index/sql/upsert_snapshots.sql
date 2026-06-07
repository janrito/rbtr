INSERT OR REPLACE INTO file_snapshots
SELECT
  repo_id,
  commit_sha,
  file_path,
  blob_sha,
  detected_language
FROM _stg
