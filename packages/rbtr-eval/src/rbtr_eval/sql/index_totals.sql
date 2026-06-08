SELECT
  COUNT(*) AS total_chunks,
  (SELECT COUNT(*) FROM edges) AS total_edges
FROM chunks
