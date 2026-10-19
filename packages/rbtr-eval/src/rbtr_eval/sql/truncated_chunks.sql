-- sqlfluff:templater:placeholder:repo_id:1
SELECT DISTINCT
  fs.file_path,
  c.scope,
  c.name,
  c.line_start,
  c.line_end,
  -- The eval's name for the column, so the frame joins onto a query
  -- without being renamed on arrival.
  c.kind AS symbol_kind
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_language = fs.detected_language
WHERE c.embedding_truncated AND fs.repo_id = $repo_id
