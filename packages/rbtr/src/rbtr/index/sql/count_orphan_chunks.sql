SELECT count(*) AS n
FROM chunks
WHERE repo_id = ?
  AND NOT EXISTS (
    SELECT 1
    FROM file_snapshots AS fs
    WHERE
      fs.repo_id = chunks.repo_id
      AND fs.blob_sha = chunks.blob_sha
      AND fs.file_path = chunks.file_path
  )
