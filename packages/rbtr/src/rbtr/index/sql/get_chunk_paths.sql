SELECT
  c.id,
  c.file_path
FROM chunks AS c
INNER JOIN _repo_refs AS rr ON c.repo_id = rr.repo_id
WHERE c.id IN (SELECT unnest($chunk_ids::text []))
