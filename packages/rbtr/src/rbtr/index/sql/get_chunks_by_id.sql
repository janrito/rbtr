-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
-- Fetch named chunks at a snapshot, joined to `_chunk_ids`.
-- One row per chunk: the ORDER BY picks which location represents
-- content that sits at several paths.
-- Chunks are matched by id, so the read stays within the columns it
-- projects.  The caller resolves its work list once and then fetches
-- each page through here.
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
INNER JOIN _chunk_ids AS w
  ON c.id = w.id
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_language = fs.detected_language
WHERE
  fs.repo_id = $repo_id
  AND fs.snapshot_sha = $snapshot_sha
ORDER BY c.id ASC, fs.file_path ASC, c.line_start ASC
