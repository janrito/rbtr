-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
-- The ids of every chunk at this snapshot still lacking a vector.
-- One row per chunk: embedding writes by chunk id, so content sitting at
-- several paths is one unit of work.
-- Reading `chunks.embedding` is most of the cost here, so this one pass
-- resolves the whole work list and the pages are fetched by id.
SELECT DISTINCT c.id
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_language = fs.detected_language
WHERE
  fs.repo_id = $repo_id
  AND fs.snapshot_sha = $snapshot_sha
  AND c.embedding IS NULL
ORDER BY c.id ASC
