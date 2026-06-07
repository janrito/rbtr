SELECT
  COUNT(*) AS total_chunks,
  COUNT(*) FILTER (embedding IS NOT NULL) AS with_embedding,
  COUNT(*) FILTER (embedding_truncated) AS truncated
FROM chunks
