UPDATE chunks
SET
  embedding = _emb_stg.embedding,
  embedding_truncated = _emb_stg.embedding_truncated
FROM _emb_stg
WHERE chunks.id = _emb_stg.id
