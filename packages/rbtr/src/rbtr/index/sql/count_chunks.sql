SELECT count(*) AS n
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.repo_id = fs.repo_id
    AND c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
WHERE
  fs.repo_id = ?
  AND fs.commit_sha = ?
