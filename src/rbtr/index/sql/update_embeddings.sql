UPDATE chunks
SET embedding = _emb_stg.embedding
FROM _emb_stg
WHERE chunks.id = _emb_stg.id
