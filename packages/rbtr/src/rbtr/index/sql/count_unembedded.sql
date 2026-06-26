-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:commit_sha:'abc'
SELECT count(*) AS n
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
WHERE
  fs.repo_id = $repo_id
  AND fs.commit_sha = $commit_sha
  AND c.embedding IS NULL
