-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
-- Each file of one snapshot and the blob it holds.
SELECT
  file_path,
  blob_sha
FROM file_snapshots
WHERE
  repo_id = $repo_id
  AND snapshot_sha = $snapshot_sha
ORDER BY file_path
