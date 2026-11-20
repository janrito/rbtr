-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
-- Each located chunk of a snapshot, with only the columns edge inference reads.
SELECT
  c.id,
  fs.file_path,
  c.line_start,
  c.kind,
  c.name,
  c.language,
  c.metadata.module,
  c.metadata.names,
  c.metadata.language_hint
FROM file_snapshots AS fs
INNER JOIN chunks AS c
  ON fs.blob_sha = c.blob_sha AND fs.detected_language = c.file_language
WHERE fs.repo_id = $repo_id AND fs.snapshot_sha = $snapshot_sha
ORDER BY fs.file_path, c.line_start, c.line_end, c.kind, c.name, c.id
