SELECT
  r.path,
  (
    SELECT COUNT(*) FROM chunks AS c
    WHERE c.repo_id = r.id
  ) AS chunks,
  (
    SELECT COUNT(*) FROM chunks AS c
    WHERE
      c.repo_id = r.id
      AND c.embedding IS NOT NULL
  ) AS embedded,
  (
    SELECT COUNT(*) FROM chunks AS c
    WHERE
      c.repo_id = r.id
      AND c.embedding_truncated
  ) AS truncated
FROM repos AS r
ORDER BY r.path
