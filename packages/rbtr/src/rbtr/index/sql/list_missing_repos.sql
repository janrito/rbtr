SELECT
  id,
  path,
  missing_since
FROM repos
WHERE missing_since IS NOT NULL
