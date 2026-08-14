-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
-- One row per chunk, not per location: embedding writes by chunk id, so a
-- chunk whose content sits at several paths is one unit of work.  The
-- ORDER BY picks which location represents it.
SELECT DISTINCT ON (c.id)
  c.id,
  fs.repo_id,
  c.blob_sha,
  fs.file_path,
  c.kind,
  c.name,
  c.scope,
  c.language,
  c.file_language,
  c.content,
  c.line_start,
  c.line_end,
  c.metadata,
  FALSE AS has_embedding
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_language = fs.detected_language
WHERE
  fs.repo_id = $repo_id
  AND fs.snapshot_sha = $snapshot_sha
  AND c.embedding IS NULL
ORDER BY fs.file_path, c.line_start
LIMIT $max_rows
