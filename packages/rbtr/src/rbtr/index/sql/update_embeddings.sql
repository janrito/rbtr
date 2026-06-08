UPDATE chunks
SET
  embedding = _emb_stg.embedding,
  embedding_truncated = _emb_stg.embedding_truncated
FROM _emb_stg
WHERE chunks.repo_id = $repo_id AND chunks.id = _emb_stg.id
