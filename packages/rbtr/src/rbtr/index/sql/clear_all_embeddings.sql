UPDATE chunks
SET embedding = NULL
WHERE embedding IS NOT NULL
