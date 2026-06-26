SELECT count(*) AS n
FROM chunks
WHERE NOT EXISTS (
  SELECT 1
  FROM file_snapshots AS fs
  WHERE
    fs.blob_sha = chunks.blob_sha
    AND fs.file_path = chunks.file_path
)
