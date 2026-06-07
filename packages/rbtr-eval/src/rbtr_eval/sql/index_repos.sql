SELECT
  r.path,
  (
    SELECT COUNT(*) FROM chunks AS c
    WHERE c.repo_id = r.id
  ) AS chunks,
  (
    SELECT COUNT(*) FROM edges AS e
    WHERE e.repo_id = r.id
  ) AS edges
FROM repos AS r
ORDER BY r.path
