-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
SELECT count(*) AS n
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_language = fs.detected_language
WHERE
  fs.repo_id = $repo_id
  AND fs.snapshot_sha = $snapshot_sha
  AND c.embedding IS NULL
