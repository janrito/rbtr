SELECT
  slug,
  COUNT(*) AS n_concepts,
  COUNT(DISTINCT file_path) AS n_files
FROM concept_stg
GROUP BY slug
ORDER BY slug
