SELECT
  c.name,
  c.kind,
  c.file_path,
  c.line_start,
  e.kind AS edge
FROM edges AS e
INNER JOIN _snapshot_refs AS rr
  ON e.repo_id = rr.repo_id AND e.snapshot_sha = rr.snapshot_sha
INNER JOIN chunks AS c
  ON e.source_id = c.id
WHERE e.target_id IN (SELECT unnest($target_ids::text []))
ORDER BY c.file_path, c.line_start
