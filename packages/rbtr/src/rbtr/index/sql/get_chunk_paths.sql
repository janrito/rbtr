-- Chunk `id` is globally unique (content-addressed), so a chunk
-- maps to exactly one file_path regardless of repo — no repo scope
-- is needed here.
SELECT
  c.id,
  c.file_path
FROM chunks AS c
WHERE c.id IN (SELECT unnest($chunk_ids::text []))
