-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:commit_sha:'abc'
-- sqlfluff:templater:placeholder:file_path:'src/main.py'
-- sqlfluff:templater:placeholder:kind:'function'
-- sqlfluff:templater:placeholder:name:'main'
SELECT
  c.id,
  fs.repo_id,
  c.blob_sha,
  c.file_path,
  c.kind,
  c.name,
  c.scope,
  c.language,
  c.content,
  c.line_start,
  c.line_end,
  c.metadata,
  c.embedding IS NOT NULL AS has_embedding
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_path = fs.file_path
WHERE
  fs.repo_id = $repo_id
  AND fs.commit_sha = $commit_sha
  AND ($file_path IS NULL OR fs.file_path = $file_path)
  AND ($kind IS NULL OR c.kind = $kind)
  AND ($name IS NULL OR c.name = $name)
ORDER BY c.file_path, c.line_start
