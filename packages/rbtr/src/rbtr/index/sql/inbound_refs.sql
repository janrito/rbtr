SELECT
  c.name,
  c.kind,
  c.file_path,
  c.line_start,
  e.kind AS edge
FROM edges AS e
INNER JOIN _repo_refs AS rr
  ON e.repo_id = rr.repo_id AND e.commit_sha = rr.commit_sha
INNER JOIN chunks AS c
  ON e.source_id = c.id
WHERE e.target_id IN (SELECT unnest($target_ids::text []))
ORDER BY c.file_path, c.line_start
