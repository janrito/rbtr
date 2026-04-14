INSERT OR REPLACE INTO file_snapshots
SELECT
  commit_sha,
  file_path,
  blob_sha
FROM _stg
